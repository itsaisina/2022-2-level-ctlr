"""
Tests for validation dataset of raw texts
"""
import json
import pathlib
import shutil

import pytest

from lab_5_scrapper.tests.s2_1_crawler_config_test import ExtendedTestCase
from lab_6_pipeline.pipeline import (CorpusManager, EmptyDirectoryError,
                                     InconsistentDatasetError)

print("Stage 2A: Validating Assets Path")
print("Starting tests with received assets folder")


def generate_test_directory(directory, raw_n=5, meta_n=5,
                            raw_empty=False, corrupted_filename=False):
    """
    An auxiliary function for creating different kind of
    directories to test dataset validator implementation
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()

    # create n raw files
    for index in range(raw_n):
        filename = f"{index + 1}_raw.txt"
        with open(directory / filename, 'w', encoding="utf-8") as file:
            text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, " \
                   "sed do eiusmod tempor incididunt ut labore et dolore " \
                   "magna aliqua. Ut enim ad minim veniam, quis nostrud " \
                   "exercitation ullamco laboris nisi ut aliquip ex ea " \
                   "commodo consequat."
            if not raw_empty:
                file.write(text)

        if corrupted_filename:
            with open(directory / f"{index + 1}_corrupted.txt", 'w', encoding="utf-8") as file:
                text = "Lorem "
                file.write(text)

    # create m meta files
    for index in range(meta_n):
        meta_dummy = {'id': 0,
                      'url': 'https://vja.ruslang.ru/ru/archive/2021-3/7-25',
                      'author': 'С.В. Князев'}
        filename = f"{index + 1}_meta.json"
        with (directory / filename).open("w", encoding='utf-8') as file:
            json.dump(meta_dummy, file, sort_keys=False,
                      indent=4, ensure_ascii=False, separators=(',', ': '))
    return directory


class PipelinePathCheck(ExtendedTestCase):
    """
    Tests for pipeline behavior relating different path input
    """

    empty = pathlib.Path('empty')
    broken_id = pathlib.Path('broken_id')
    imbalanced = pathlib.Path('imbalanced')
    empty_raw = pathlib.Path('empty_raw')
    normal = pathlib.Path('normal')
    corrupted_filename = pathlib.Path('corrupted_filename')
    filepath = pathlib.Path('filepath.txt')

    @classmethod
    def setUpClass(cls) -> None:
        generate_test_directory(cls.empty, raw_n=0, meta_n=0)
        generate_test_directory(cls.broken_id)
        (cls.broken_id / '1_raw.txt').unlink()
        generate_test_directory(cls.imbalanced, raw_n=3, meta_n=2)
        generate_test_directory(cls.empty_raw, raw_empty=True)
        generate_test_directory(cls.normal)
        generate_test_directory(cls.corrupted_filename, corrupted_filename=True)
        cls.filepath.open('w', encoding='utf-8').close()

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipe_fails_given_non_existent_path(self):
        """
        Ensure that pipeline raises an error when given invalid path
        """
        non_existent_path = pathlib.Path("non_existent_path")
        error_message = "Checking that scrapper can handle " \
                        "not existing assets paths failed."
        self.assertRaisesWithMessage(error_message,
                                     FileNotFoundError,
                                     CorpusManager,
                                     non_existent_path)

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipeline_fails_given_filepath(self):
        """
        Ensure that pipeline raises an error when given non-existing directory
        """
        error_message = "Checking that pipeline fails " \
                        "if given not a directory path."
        self.assertRaisesWithMessage(error_message,
                                     NotADirectoryError,
                                     CorpusManager,
                                     self.filepath)

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipe_fails_given_empty_directory(self):
        """
        Ensure that pipeline raises an error when given empty directory
        """
        error_message = "Checking that empty directories " \
                        "cannot be processed."
        self.assertRaisesWithMessage(error_message,
                                     EmptyDirectoryError,
                                     CorpusManager,
                                     self.empty)

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipe_fails_given_inconsistent_dataset(self):
        """
        Check consistent numbering
        """
        error_message = "Checking that pipeline does not accept " \
                        "dataset with inconsistent numeration"
        self.assertRaisesWithMessage(error_message,
                                     InconsistentDatasetError,
                                     CorpusManager,
                                     self.broken_id)

    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipe_fails_given_imbalanced_dataset(self):
        """
        Check consistent numbering among meta and text files
        """
        error_message = "Checking that pipeline does not accept " \
                        "dataset with uneven numbers of meta and text files"
        self.assertRaisesWithMessage(error_message,
                                     InconsistentDatasetError,
                                     CorpusManager,
                                     self.imbalanced)

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipe_fails_given_dataset_with_empty_texts(self):
        """
        Check that pipeline does not work with empty files
        """
        error_message = "Checking that pipeline does not accept " \
                        "dataset with empty text files"
        self.assertRaisesWithMessage(error_message,
                                     InconsistentDatasetError,
                                     CorpusManager,
                                     self.empty_raw)

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_1_dataset_sanity_checks
    @pytest.mark.lab_6_pipeline
    def test_pipe_skips_other_files(self):
        """
        Check that pipeline works with only _raw.txt and _meta.json
        """
        error_message = "Checking that pipeline does not accept " \
                        "dataset with files with corrupted names"
        self.assertIsInstance(CorpusManager(self.corrupted_filename),
                              CorpusManager, msg=error_message)

    @classmethod
    def tearDownClass(cls) -> None:
        for path in [cls.empty, cls.broken_id, cls.imbalanced,
                     cls.empty_raw, cls.normal, cls.corrupted_filename]:
            shutil.rmtree(path)
        cls.filepath.unlink()
