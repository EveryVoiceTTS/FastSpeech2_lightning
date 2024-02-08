#!/usr/bin/env python

"""
If you've installed `everyvoice` and would like to run this unittest:
python -m unittest everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.tests.test_lookuptable
"""

from unittest import TestCase, main


class LookupTableTest(TestCase):
    def test_lookuptable_definitions(self):
        """
        To keep fs2's CLI --help speedy, we decided to redefine LookupTable.
        This testcase makes sure the two definitions stay aligned.
        """

        from everyvoice.text.lookups import LookupTable as lt1

        try:
            from fs2.type_definitions import LookupTable as lt2
        except ImportError:
            from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
                LookupTable as lt2,
            )

        self.assertEqual(lt1, lt2)


if __name__ == "__main__":
    main()