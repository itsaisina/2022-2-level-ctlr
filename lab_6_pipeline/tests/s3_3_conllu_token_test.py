"""
Tests for ConlluToken functionality
"""
import unittest

import pytest

from lab_6_pipeline.pipeline import (ConlluSentence, ConlluToken,
                                     MorphologicalTokenDTO)


class ConlluTokenTestMinimal(unittest.TestCase):
    """
    Tests for ConlluToken mark 4 realization
    """
    def setUp(self) -> None:
        self.token = ConlluToken('Original_token')

    def test_conllu_token_instantiation(self):
        """
        Ensure that ConlluToken instance is instantiated correctly
        """
        new_token = ConlluToken('оригинальное слово')
        self.assertTrue(hasattr(new_token, '_text'),
            "ConlluToken instance must possess the following arguments: '_text'")

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_3_conllu_token_checks
    @pytest.mark.lab_6_pipeline
    def test_get_cleaned(self) -> None:
        """
        Ensure that clean representation of a token is appropriate
        """
        good_token = ConlluToken('МАМА')
        self.assertEqual(good_token.get_cleaned(), 'мама')
        num_token = ConlluToken('2002')
        self.assertEqual(num_token.get_cleaned(), '2002')
        bad_token = ConlluToken(',')
        self.assertEqual(bad_token.get_cleaned(), '')


class ConlluTokenTest(unittest.TestCase):
    """
    Tests for ConlluToken realization
    """

    def setUp(self) -> None:
        self.token = ConlluToken('Original_token')

        morphological_parameters = MorphologicalTokenDTO()
        self.token.set_morphological_parameters(morphological_parameters)

    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_3_conllu_token_checks
    @pytest.mark.lab_6_pipeline
    def test_conllu_token_instantiation(self) -> None:
        """
        Ensure that ConlluToken instance is instantiated correctly
        """
        new_token = ConlluToken('оригинальное слово')
        attrs = ['_text', '_morphological_parameters']
        self.assertTrue(all((
            hasattr(new_token, attrs[0]),
            hasattr(new_token, attrs[1]))),
            f"ConlluToken instance must possess the following arguments: {', '.join(attrs)}")

        morphological_params = new_token.get_morphological_parameters()

        all_parameters = (morphological_params.tags, morphological_params.pos,
                          morphological_params.lemma)

        self.assertFalse(any((
            all_parameters)),
            f"{', '.join(all_parameters[1:])} "
            f"fields of ConlluToken instance must initially be empty")


class ConlluSentenceMinimalTest(unittest.TestCase):
    """
    Tests for ConlluSentence mark 4 realization
    """
    def setUp(self) -> None:
        self.tokens = [ConlluToken('Мама'), ConlluToken(','),
                       ConlluToken('мыла'), ConlluToken('10'), ConlluToken('.')]
        self.sentence = ConlluSentence(1, 'Мама, мыла 10.', self.tokens)

    @pytest.mark.mark4
    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_3_conllu_token_checks
    @pytest.mark.lab_6_pipeline
    def test_get_cleaned(self):
        """
        Tests for get_cleaned method
        """
        self.assertEqual(self.sentence.get_cleaned_sentence(), 'мама мыла 10')


class ConlluSentenceTest(unittest.TestCase):
    """
    Tests for ConlluSentence realization
    """

    def setUp(self) -> None:
        self.tokens = [ConlluToken('Мама'), ConlluToken('мыла')]
        self.sentence = ConlluSentence(1, 'Мама мыла', self.tokens)

    @pytest.mark.mark10
    @pytest.mark.stage_3_3_conllu_token_checks
    @pytest.mark.lab_6_pipeline
    def test_get_tokens(self) -> None:
        """
        Test for get_tokens method
        """
        self.assertEqual(self.sentence.get_tokens(), self.sentence._tokens)  # pylint: disable=protected-access


class MorphologicalParametersTest(unittest.TestCase):
    """
    Tests for morphological token parameters realization
    """

    def setUp(self) -> None:
        self.morphological_parameters = MorphologicalTokenDTO()

    @pytest.mark.mark6
    @pytest.mark.mark8
    @pytest.mark.mark10
    @pytest.mark.stage_3_3_conllu_token_checks
    @pytest.mark.lab_6_pipeline
    def test_morph_params_instantiation(self) -> None:
        """
        Ensure that morphological token parameters instance is instantiated correctly
        """
        attrs = ['lemma', 'pos', 'tags']
        needed_attrs = []
        self.assertTrue(all((True if hasattr(self.morphological_parameters, attr)
                             else needed_attrs.append(attr) for attr in attrs)),
                        f"Morphological token parameters instance"
                        f" must possess the following arguments: {', '.join(attrs)}")
