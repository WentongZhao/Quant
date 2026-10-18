"""Regression checks for precision routing and trustworthy evaluation records."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from evaluation.cli import parse_args
from evaluation.dispatch import libero_config, run_evaluation, RUNNERS


class ConfigurationTests(unittest.TestCase):
    def test_fp_needs_no_calibration_and_has_no_pseudo_quant(self):
        args = parse_args(['--checkpoint', 'model', '--mode', 'fp'])
        config = libero_config(args)
        self.assertEqual(config['data'], '')
        self.assertFalse(config['pseudo_quant'])
        self.assertEqual(config['eval_mode'], 'libero_fp')

    def test_cli_overrides_yaml_and_relative_paths_follow_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.yaml'
            path.write_text('checkpoint: model\nnum_trials_per_task: 2\noutput_dir: ./out\n')
            args = parse_args(['--config', str(path), '--num_trials_per_task', '3'])
            self.assertEqual(args.num_trials_per_task, 3)
            self.assertEqual(args.output_dir, str((Path(tmp) / 'out').resolve()))

    def test_ambiguous_or_unknown_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.yaml'
            for extra in ('sq_w_asym: "false"', 'unknown_option: true'):
                path.write_text('checkpoint: model\n' + extra + '\n')
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(['--config', str(path)])

    def test_invalid_precision_requests_fail_early(self):
        for flags in (['--mode', 'fp', '--run_process'], ['--mode', 'quant'],
                      ['--mode', 'quant', '--data', 'data', '--k_bit', '8'],
                      ['--eval_task_ids', '0', '0']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--checkpoint', 'model'] + flags)

    def test_quant_config_enforces_requested_symmetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            scale = Path(tmp) / 'scale.pt'
            scale.touch()
            args = parse_args(['--checkpoint', 'model', '--mode', 'quant', '--data', tmp,
                               '--scale_path', str(scale), '--w_bit', '4', '--a_bit', '4'])
            cfg = libero_config(args)
            self.assertTrue(cfg['pseudo_quant'])
            self.assertFalse(cfg['sq_w_asym'])
            self.assertTrue(cfg['sq_a_asym'])
            self.assertEqual((cfg['w_bit'], cfg['a_bit']), (4, 4))

    def test_pending_routes_are_not_advertised(self):
        self.assertNotIn(('openvla_oft', 'simplerenv', 'quant'), RUNNERS)
        args = parse_args(['--checkpoint', 'model', '--model_backend', 'unimplemented'])
        with self.assertRaisesRegex(ValueError, 'Combination not wired'):
            run_evaluation(args)

    def test_dispatch_isolates_rollouts_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--checkpoint', 'model', '--output_dir', tmp])
            with mock.patch('evaluation.dispatch.subprocess.run', return_value=SimpleNamespace(returncode=2)) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_evaluation(args)
                self.assertEqual(run.call_args.kwargs['cwd'], str(Path(tmp).resolve()))
                self.assertEqual(Path(run.call_args.kwargs['env']['PYTHONPATH']).name, 'Quant')
            record = json.loads((Path(tmp) / 'run.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['exit_code'], 2)
            with self.assertRaises(FileExistsError):
                run_evaluation(args)


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main_quant
        cls.backend = main_quant

    def args(self, mode='libero_fp'):
        with mock.patch.object(sys, 'argv', ['main_quant.py', '--model', 'model', '--eval_mode', mode]):
            args = self.backend.parse_quant_args()
        return args

    def invoke_mocked(self, args):
        original = mock.MagicMock(llm_dim=16)
        quantized = mock.MagicMock(llm_dim=16)
        with contextlib.ExitStack() as stack:
            for name in ('get_processor', 'get_proprio_projector', 'get_action_head', 'check_unnorm_key', '_set_reproducibility'):
                stack.enter_context(mock.patch.object(self.backend, name))
            stack.enter_context(mock.patch.object(self.backend, 'get_vla', return_value=original))
            calibrate = stack.enter_context(mock.patch.object(self.backend, 'get_libero_calib_dataset', return_value=([], {})))
            quantize = stack.enter_context(mock.patch.object(self.backend, 'qwrapper', return_value=quantized))
            evaluate = stack.enter_context(mock.patch.object(self.backend, '_evaluate_libero'))
            self.backend.cli_quant_single(args)
        return original, quantized, calibrate, quantize, evaluate

    def test_fp_bypasses_calibration_and_quantization(self):
        args = self.args()
        original, _, calibrate, quantize, evaluate = self.invoke_mocked(args)
        calibrate.assert_not_called()
        quantize.assert_not_called()
        self.assertIs(evaluate.call_args.args[2], original)
        self.assertFalse(evaluate.call_args.kwargs['quantized'])
        self.assertTrue(evaluate.call_args.args[1].fail_on_episode_error)

    def test_checkpoint_source_lookup_is_independent_of_launch_directory(self):
        from experiments.robot import openvla_utils
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(openvla_utils.os, 'walk', return_value=[]) as walk:
            openvla_utils.check_model_logic_mismatch(tmp)
        expected = Path(openvla_utils.__file__).resolve().parents[2] / 'prismatic'
        self.assertEqual(walk.call_args.args[0], expected)
        self.assertTrue(expected.is_dir())

    def test_quant_uses_model_returned_by_quantizer(self):
        args = self.args('libero_quant')
        args.data, args.method, args.pseudo_quant = 'calibration', 'smoothquant', True
        _, quantized, calibrate, quantize, evaluate = self.invoke_mocked(args)
        calibrate.assert_called_once()
        quantize.assert_called_once()
        self.assertIs(evaluate.call_args.args[2], quantized)
        self.assertTrue(evaluate.call_args.kwargs['quantized'])

    def test_invalid_backend_requests_do_not_load_model(self):
        for mode, pseudo, process in [('libero_fp', True, False), ('libero_fp', False, True),
                                      ('libero_quant', False, False)]:
            args = self.args(mode)
            args.pseudo_quant, args.run_process = pseudo, process
            args.method, args.data = 'smoothquant', 'data'
            with self.subTest(mode=mode, pseudo=pseudo, process=process), mock.patch.object(self.backend, 'get_vla') as load:
                with self.assertRaises(ValueError):
                    self.backend.cli_quant_single(args)
                load.assert_not_called()


class MetricTests(unittest.TestCase):
    def run_fake_suite(self, result_path, task_results):
        import evaluation.libero as worker
        args = SimpleNamespace(eval_task_ids=[0, 2], results_json=str(result_path), model='model')
        cfg = SimpleNamespace(seed=7, task_suite_name='test', num_trials_per_task=2, unnorm_key='test',
                              num_images_in_input=2, use_proprio=True, num_open_loop_steps=8)
        vla = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype='bfloat16')]),
                              language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation='flash_attention_2')))
        suite = SimpleNamespace(n_tasks=3, get_task=lambda t: SimpleNamespace(language=f'task {t}'))
        log_file = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(worker, 'validate_config'))
            stack.enter_context(mock.patch.object(worker, 'set_seed_everywhere'))
            stack.enter_context(mock.patch.object(worker, 'get_image_resize_size', return_value=224))
            stack.enter_context(mock.patch.object(worker.benchmark, 'get_benchmark_dict', return_value={'test': lambda: suite}))
            stack.enter_context(mock.patch.object(worker, 'setup_logging', return_value=(log_file, 'log', 'run')))
            task = stack.enter_context(mock.patch.object(worker, 'run_task', side_effect=task_results))
            worker.evaluate_libero(args, cfg, vla, None, None, None, quantized=False)
        self.assertTrue(log_file.closed)
        return task

    def test_task_deltas_and_total_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / 'result.json'
            task = self.run_fake_suite(result_path, [(2, 1), (4, 3)])
            result = json.loads(result_path.read_text())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['total_episodes'], 4)
            self.assertEqual(result['success_rate'], 0.75)
            self.assertEqual([t['successes'] for t in result['tasks']], [1, 2])
            self.assertIsNone(result['quantization'])
            self.assertEqual([c.args[2] for c in task.call_args_list], [0, 2])
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_runtime_error_marks_failed_with_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / 'result.json'
            with self.assertRaisesRegex(RuntimeError, 'simulator failed'):
                self.run_fake_suite(result_path, [(2, 1), RuntimeError('simulator failed')])
            result = json.loads(result_path.read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['total_episodes'], 2)
            self.assertIn('simulator failed', result['error'])


if __name__ == '__main__':
    unittest.main()
