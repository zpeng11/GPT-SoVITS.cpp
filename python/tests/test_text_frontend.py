import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_frontend import TextFrontend
from text_frontend.errors import TextFrontendError
from text_frontend.symbols import SYMBOLS
from text_frontend.tokenizer import WordPieceTokenizer
from text_frontend.layout import pack_bert_features_for_ggml, phoneme_ids_int32


class TextFrontendTests(unittest.TestCase):
    def require_english_resources(self):
        import importlib.util
        from text_frontend.english_model import model_directory
        from text_frontend.english_resources import tagger_directory
        for name in ('numpy', 'nltk', 'wordsegment'):
            if importlib.util.find_spec(name) is None:
                self.skipTest(f'English integration requires {name}')
        try:
            model_directory()
            tagger_directory()
        except TextFrontendError as exc:
            self.skipTest(str(exc))

    def test_symbols(self):
        self.assertEqual(len(SYMBOLS), 732)
        self.assertEqual(len(set(SYMBOLS)), 732)

    def test_chinese_alignment(self):
        result = TextFrontend().process('今天是2026年10月20日，重庆银行利率3.5%。', 'zh')
        self.assertIn('百分之三点五', result.normalized_text)
        self.assertEqual(len(result.word2ph), len(result.normalized_text))
        self.assertEqual(sum(result.word2ph), result.phoneme_count)
        self.assertTrue(all(0 <= value < 732 for value in result.phoneme_ids))

    def test_chinese_grouped_numbers(self):
        frontend = TextFrontend()
        cases = {
            '金额1,234.56元。': '金额一千二百三十四点五六元.',
            '金额1,234元。': '金额一千二百三十四元.',
            '金额1,234,567.89元。': '金额一百二十三万四千五百六十七点八九元.',
            '金额-1,234.56元。': '金额负一千二百三十四点五六元.',
            '金额１,２３４.５６元。': '金额一千二百三十四点五六元.',
            '增长1,234.5%。': '增长百分之一千二百三十四点五.',
            '1,234,结束': '一千二百三十四,结束',
            '先说一,再说二。': '先说一,再说二.',
            '报价,1,234元。': '报价,一千二百三十四元.',
            '余额,-1,234元。': '余额,负一千二百三十四元.',
            '温度1,234℃。': '温度一千二百三十四度.',
            '温度-1,234℃。': '温度零下一千二百三十四度.',
            '长度1,234mm~2,345mm。': '长度一千二百三十四毫米至二千三百四十五毫米.',
            '长度1,234 mm ~ 2,345 mm。': '长度一千二百三十四毫米至二千三百四十五毫米.',
            '数量1,234~2,345。': '数量一千二百三十四到二千三百四十五.',
            '数量1,234+2,345。': '数量一千二百三十四加二千三百四十五.',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = frontend.process(text, 'zh')
                self.assertEqual(result.normalized_text, expected)
                self.assertEqual(len(result.word2ph), len(expected))
                self.assertEqual(sum(result.word2ph), result.phoneme_count)
        from text_frontend.zh_normalization.text_normlization import TextNormalizer
        normalizer = TextNormalizer()
        for text in ('1,23元。', '1,234,56元。', '12,34,567元。'):
            with self.subTest(malformed=text):
                self.assertEqual(normalizer.normalize(text), [
                    normalizer.normalize_sentence(part) for part in normalizer._split(text)
                ])

    def test_chinese_measurement_units_longest_match(self):
        from text_frontend.zh_normalization.quantifier import measure_dict, replace_measure
        from text_frontend.normalize_zh import normalize_zh
        for unit, spoken in measure_dict.items():
            with self.subTest(unit=unit):
                self.assertEqual(replace_measure('5' + unit), '5' + spoken)
        self.assertEqual(normalize_zh('5mm'), '五毫米')
        self.assertEqual(normalize_zh('长度5mm，体积5cm3。'), '长度五毫米,体积五立方厘米.')
        result = TextFrontend().process('长度5mm，体积5cm3。', 'zh')
        self.assertEqual(sum(result.word2ph), result.phoneme_count)
        self.assertEqual(len(result.word2ph), len(result.normalized_text))

    def test_unknown_chinese_characters_keep_alignment(self):
        import contextlib
        import io
        from unittest.mock import patch
        from text_frontend.errors import UnknownWordError
        cases = ('𠮷野家', '你好𠮷𠮷龘。', '𠮷𠮷', '不𠮷', '一𠮷', '𠮷儿', '好𠮷好', '𠮷哈哈')
        for fallback in (False, True):
            for text in cases:
                with self.subTest(text=text, character_fallback=fallback):
                    with contextlib.ExitStack() as stack:
                        output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        if fallback:
                            stack.enter_context(patch('text_frontend.normalize_zh._segment_with_jieba', return_value=None))
                        result = TextFrontend().process(text, 'zh')
                        self.assertEqual(output.getvalue(), '')
                        self.assertEqual(result.phones.count('UNK'), text.count('𠮷'))
                        self.assertEqual(len(result.word2ph), len(result.normalized_text))
                        self.assertEqual(sum(result.word2ph), result.phoneme_count)
                        self.assertTrue(any('UNK' in warning for warning in result.warnings))
                        with self.assertRaises(UnknownWordError):
                            TextFrontend(strict=True).process(text, 'zh')
        result = TextFrontend().process('好𠮷好', 'zh')
        self.assertEqual(result.phones, ['h', 'ao3', 'UNK', 'h', 'ao3'])

    def test_unknown_chinese_cli_json(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        for text in ('𠮷野家', '你好𠮷𠮷龘。', '𠮷儿'):
            for strict in (False, True):
                with self.subTest(text=text, strict=strict):
                    args = [sys.executable, '-m', 'text_frontend', '--language', 'zh', '--text', text]
                    if strict:
                        args.append('--strict')
                    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=20)
                    payload = json.loads(proc.stdout)
                    self.assertEqual(proc.returncode, 2 if strict else 0, proc.stderr)
                    if strict:
                        self.assertEqual(payload['error']['type'], 'UnknownWordError')
                    else:
                        self.assertIn('UNK', payload['phones'])

    def test_english(self):
        self.require_english_resources()
        result = TextFrontend().process('Hello world!', 'en')
        self.assertGreater(result.phoneme_count, 0)
        self.assertIsNone(result.word2ph)
        self.assertFalse(result.unknown_words)

    def test_mixed_offsets(self):
        self.require_english_resources()
        result = TextFrontend().process('你好，Hello world! 今天见。')
        self.assertEqual([s.language for s in result.segments], ['zh', 'en', 'zh'])
        self.assertEqual(result.segment_phoneme_offsets()[-1], result.phoneme_count)
        self.assertEqual(''.join(s.normalized_text for s in result.segments), result.normalized_text)
        for segment in result.segments:
            self.assertEqual(result.phoneme_ids[segment.phoneme_offset:segment.phoneme_offset + len(segment.phoneme_ids)], segment.phoneme_ids)

    def test_invalid_inputs(self):
        for text in ('', '   ', 'こんにちは'):
            with self.subTest(text=text), self.assertRaises(TextFrontendError):
                TextFrontend().process(text)
        with self.assertRaises(TextFrontendError):
                TextFrontend(strict=True, english_backend='basic').process('zzqxxzz', 'en')
        with self.assertRaises(TextFrontendError):
            TextFrontend(tokenizer_path='/nonexistent/tokenizer.json')

    def test_tokenizer_specials(self):
        tokenizer = WordPieceTokenizer({'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, '你': 3, '好': 4})
        result = tokenizer.encode('你好')
        self.assertEqual(result.input_ids, [1, 3, 4, 2])
        self.assertEqual(result.body_token_ids, [3, 4])

    def test_review_regressions(self):
        from text_frontend.normalize_en import normalize_en
        from text_frontend.normalize_zh import replace_punctuation
        self.assertEqual(normalize_en('I have 2.'), 'I have two.')
        self.assertIn('four', normalize_en('Use GPT4 today'))
        self.assertIn('euros', normalize_en('Pay €5 now'))
        self.assertIn('and', normalize_en('Fish & chips'))
        self.assertIn('one thousand two hundred thirty-four point five six', normalize_en('I paid 1,234.56 dollars.'))
        self.assertEqual(replace_punctuation('你好𠮷龘'), '你好𠮷龘')
        tokenizer = WordPieceTokenizer({'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, '你': 3, '好': 4})
        self.assertEqual(tokenizer.encode('你好').token_char_spans, [None, (0, 1), (1, 2), None])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokenizer.json'
            path.write_text(json.dumps({'model': {'type': 'WordPiece', 'vocab': tokenizer.vocab}, 'normalizer': {'type': 'Sequence', 'normalizers': [{'type': 'BertNormalizer'}, {'type': 'Lowercase'}]}}))
            with self.assertRaises(TextFrontendError):
                WordPieceTokenizer.from_file(path)

    def test_offline_english_failure_is_not_basic_fallback(self):
        import socket
        from unittest.mock import patch
        from text_frontend.errors import ResourceError
        with tempfile.TemporaryDirectory() as directory, patch.object(socket, 'socket', side_effect=AssertionError('network')):
            frontend = TextFrontend(nltk_data_dir=directory)
            with self.assertRaises(ResourceError) as caught:
                frontend.process('Hello world', 'en')
            self.assertIn('--prepare-english-resources', str(caught.exception))
            result = TextFrontend(english_backend='basic', nltk_data_dir=directory).process('Hello world', 'en')
            self.assertEqual(result.to_dict()['english_backend'], 'basic')
            self.assertTrue(any('explicit basic' in warning for warning in result.warnings))

    def test_package_import_is_lazy_and_network_free(self):
        code = '''
import socket, sys
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
from text_frontend import TextFrontend
TextFrontend()
assert not {'g2p_en', 'nltk', 'wordsegment', 'numpy'} & set(sys.modules)
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)

    def test_pos_pronunciation(self):
        self.require_english_resources()
        frontend = TextFrontend()
        examples = {
            'I read books every day.': ['AY1', 'R', 'IY1', 'D'],
            'I use it.': ['AY1', 'Y', 'UW1', 'Z'],
            'a complex problem': ['AH0', 'K', 'AH0', 'M', 'P', 'L', 'EH1', 'K', 'S'],
        }
        for text, prefix in examples.items():
            with self.subTest(text=text):
                self.assertEqual(frontend.process(text, 'en').phones[:len(prefix)], prefix)
        record = frontend.process('Please record the record.', 'en').phones
        self.assertEqual(record[4:10], ['R', 'IH0', 'K', 'AO1', 'R', 'D'])
        self.assertEqual(record[-6:-1], ['R', 'EH1', 'K', 'ER0', 'D'])
        result = frontend.process('quux xyzzy flibbertigibbet', 'en')
        self.assertTrue(any('numpy_g2p' in warning for warning in result.warnings))
        self.assertIsNotNone(frontend._english._predictor)
        self.assertEqual(result.phones[:3], ['K', 'AH1', 'S'])
        with self.assertRaises(TextFrontendError):
            TextFrontend(strict=True).process('quux', 'en')

    def test_normalization_oie_coverage(self):
        from text_frontend.normalize_en import normalize_en
        self.assertEqual(normalize_en('NASA'), 'N A S A')
        self.assertEqual(normalize_en('32.3$'), 'thirty-two dollars and thirty cents')
        self.assertEqual(normalize_en('1.1£'), 'one pound and ten pence')
        self.assertEqual(normalize_en('Hello!!!'), 'Hello!')

    def test_runtime_does_not_use_global_nltk_or_downloads(self):
        self.require_english_resources()
        import nltk
        import socket
        from unittest.mock import patch
        from text_frontend.english_resources import nltk_data_directory
        configured = nltk_data_directory()
        saved_path = list(nltk.data.path)
        with patch.object(socket, 'socket', side_effect=AssertionError('network')), patch.object(nltk, 'download', side_effect=AssertionError('download')), patch.object(nltk.data, 'path', ['/nonexistent/global-nltk']):
            frontend = TextFrontend(nltk_data_dir=configured)
            result = frontend.process('I read quux.', 'en')
            self.assertIn('IY1', result.phones)
            self.assertEqual(nltk.data.path, ['/nonexistent/global-nltk'])
        self.assertEqual(nltk.data.path, saved_path)
        self.assertNotIn('g2p_en', sys.modules)

    def test_numpy_predictor_matches_upstream(self):
        import ast
        import types
        import numpy as np
        from text_frontend.english_model import NumpyG2P, model_directory
        try:
            directory = model_directory()
        except TextFrontendError as exc:
            self.skipTest(str(exc))
        source = directory / 'g2p.py'
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(source.read_text(encoding='utf-8'))
        upstream = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'G2p')
        methods = [node for node in upstream.body if isinstance(node, ast.FunctionDef) and node.name in {'sigmoid', 'grucell', 'gru', 'encode', 'predict'}]
        namespace = {'np': np}
        module = ast.Module(body=methods, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        predictor = NumpyG2P(directory)
        reference = types.SimpleNamespace()
        for name in ('sigmoid', 'grucell', 'gru', 'encode', 'predict'):
            setattr(reference, name, types.MethodType(namespace[name], reference))
        from text_frontend.english_model import PHONEMES
        reference.g2idx = predictor.g2idx
        reference.idx2p = dict(enumerate(PHONEMES))
        with np.load(directory / 'checkpoint20.npz', allow_pickle=False) as checkpoint:
            for name in checkpoint.files:
                setattr(reference, name, checkpoint[name])
        for word in ('quux', 'xyzzy', 'flibbertigibbet', 'sovits', 'kubernetes', 'zpeng'):
            with self.subTest(word=word):
                self.assertEqual(predictor.predict(word), reference.predict(word))

    def test_repeated_punctuation_token_offsets(self):
        self.require_english_resources()
        frontend = TextFrontend()
        for text in ('well----known', "rock''''n", 'foo-----bar baz'):
            with self.subTest(text=text):
                result = frontend._english.g2p(text)
                self.assertTrue(result.phones)
                self.assertEqual(result.token_spans[0][0], 0)
                self.assertEqual(result.token_spans[-1][1], len(text))
                self.assertTrue(all(0 <= start < end <= len(text) for start, end, _, _ in result.token_spans))

    def test_required_hot_dictionary_and_corrupt_pos_model(self):
        import shutil
        from text_frontend.resources import Resources, DEFAULT_DATA_DIR
        from text_frontend.errors import ResourceError
        from text_frontend.english_resources import TAGGER_FILES, TAGGER_NAME, load_pos_tagger
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(DEFAULT_DATA_DIR / 'cmudict.rep', root / 'cmudict.rep')
            with self.assertRaises(ResourceError) as caught:
                Resources(root).load_english_dictionary()
            self.assertIn('engdict-hot.rep', str(caught.exception))
            tagger = root / 'taggers' / TAGGER_NAME
            tagger.mkdir(parents=True)
            for name, data in zip(TAGGER_FILES, ({}, {'I': 'PRP'}, ['PRP', 'NN'])):
                (tagger / name).write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ResourceError) as caught:
                load_pos_tagger(root)
            self.assertIn('empty perceptron weights', str(caught.exception))

    def test_oie_pronunciation_fixtures(self):
        self.require_english_resources()
        fixture = Path(__file__).parent / 'fixtures' / 'english_oie.json'
        cases = json.loads(fixture.read_text(encoding='utf-8'))['cases']
        frontend = TextFrontend()
        for case in cases:
            with self.subTest(name=case['name'], text=case['text']):
                self.assertEqual(frontend._english.g2p(case['text']).phones, case['phones'])

    def test_layout(self):
        import numpy as np
        array = np.arange(3 * 1024, dtype=np.float32).reshape(3, 1024)
        self.assertEqual(pack_bert_features_for_ggml(array, expected_t=3), array.tobytes())
        self.assertEqual(phoneme_ids_int32([1, 2]).dtype, np.int32)
        with self.assertRaises(TextFrontendError):
            pack_bert_features_for_ggml(array.T)

    def test_bert_expansion_excludes_specials(self):
        import numpy as np
        from text_frontend.layout import expand_bert_features
        tokenizer = WordPieceTokenizer({'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, '你': 3, '好': 4})
        tokens = tokenizer.encode('你好')
        hidden = np.repeat(np.arange(4, dtype=np.float32)[:, None], 1024, axis=1)
        result = expand_bert_features(hidden, tokens, [2, 2])
        self.assertEqual(result[:, 0].tolist(), [1, 1, 2, 2])
        self.assertTrue(result.flags.c_contiguous)
        with self.assertRaises(TextFrontendError):
            expand_bert_features(hidden, tokens, [2, 2, 1])

    def test_no_heavy_imports(self):
        self.require_english_resources()
        code = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'MNN', 'onnxruntime'}:
                raise AssertionError('heavy import: ' + fullname)
        if fullname == 'g2p_en' or fullname.startswith('g2p_en.'):
            raise AssertionError('downloading upstream import: ' + fullname)
sys.meta_path.insert(0, Block())
from text_frontend import TextFrontend
TextFrontend().process('你好，quux xyzzy!')
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)

    def test_cli(self):
        self.require_english_resources()
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.npz'
            proc = subprocess.run([sys.executable, '-m', 'text_frontend', '--text', '你好，Hello!', '--npz', str(path)], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            payload = json.loads(proc.stdout)
            import numpy as np
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data['phoneme_ids'].dtype, np.int32)
                self.assertEqual(data['phoneme_ids'].tolist(), payload['phoneme_ids'])


if __name__ == '__main__':
    unittest.main()
