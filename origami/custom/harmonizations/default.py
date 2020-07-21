# generic transcription harmonization schema, built for use
# with text from various sources.

{
	"channels": {
		"default": {
			"transform": "default",
			"alphabet": {
				"lowercase": "abcdefghijklmnopqrstuvwxyzßø",
				"uppercase": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",

				"ligatures": "æÆœ",
				"lowercase_diacritic": "äöüàáâåãéèëêẽôòõîĩůûùũñç",
				"uppercase_diacritic": "ÄÖÜÅÊÇÙÛÚÒ",

				"punctuation": "_-?!.,:; ",
				"quotes": "‚'",
				"brackets": "()[]<>«»",
				"slashes": "/|",
				"math": "*+=%",

				"digits": "1234567890",
				"currency": "€£",

				"symbols": "§&",
				"poetry": "⏓⏑",
				"greek": "Ἰϰχεά",
				"rotunda": "ꝛ",
				"eth": "ð"
			},
			"tests": ["default"]
		}
	},
	"tests": {
		"default": [
			("a  b c", "a b c"),
			("a.b", "a. b"),
			("a,b", "a, b"),
			("a:b", "a: b"),
			("a .b", "a. b"),
			("a ,b", "a, b"),
			("a ( b ) c", "a (b) c"),
			("a ( b ) , c", "a (b), c"),
			("a ( b ) . c", "a (b). c"),
			("a„  b ”c", "a ‚‚b'' c"),
			("a  '  b", "a ' b"),
			("a 3. 7. 14.). b", "a 3. 7. 14.). b"),
			("1 %", "1%"),
			("12 - 34", "12-34"),
			("12 -- 34", "12--34"),
			("12 , 34", "12,34")
		]
	},
	"transforms": {
		"default": [
			("unicode", "NFC"),

			# normalize dashes.
			("str", "⸗", "-"),  # in Calamari fraktur 2020 prediction
			("str", "·", "-"),  # in gt4hist prediction
			("str", "¬", "-"),  # convert double-dash to short dash. (NZZ)
			("str", "—", "―"),  # convert NZZ long dash to short dash. (NZZ)
			("str", "–", "-"),  # 1832-lenau_gedichte_00587
			("str", "―", "--"),

			# normalize visually highly similar symbols.
			("str", "Ʒ", "3"),  # in gt4hist prediction
			("str", "ʒ", "3"),  # in gt4hist prediction
			("str", "ꝗ", "q"),  # in gt4hist prediction
			("str", "ꝓ", "p"),  # in gt4hist prediction
			("str", "ꝑ", "p"),  # in tesseract prediction
			("str", ("chr", 868), "e"),  # upper e
			("str", "ꝰ", "9"),  # upper 9
			("str", "ν", "v"),


			# normalize quotes and apostrophes.
			("str", "’", "'"),
            ("str", "\"", "''"),

            ("str", "”", "''"),
            ("str", "„", "‚‚"),
            ("str", "“", "''"),

			("str", ("chr", 0x2018), "'"),  # apostrophe

			# normalize whitespace before and after quotes.
			("re", r"([^‚\s])‚‚", r"\g<1> ‚‚"),
			("re", r"‚‚\s+", "‚‚"),

			("re", r"''([^'\s])", r"'' \g<1>"),
			("re", r"\s+''", "''"),

			# ignore esoteric symbols.
			("str", "¶", ""),  # in tesseract prediction
			("str", ("chr", 0x303), ""),  # in tesseract prediction

			# normalize Fraktur-s.
			("str", "ſ", "s"),

			# expand fractions and other composite symbols.
			("str", "½", " 1/2 "),
			("str", "¼", " 1/4 "),
			("str", "¾", " 3/4 "),
			("str", "⅔", " 2/3 "),
			("str", "…", "..."),  # in tesseract prediction

			# normalize whitespace after punctuation symbols.
			("str", ":", ": "),
			("str", ";", "; "),
			("str", ".", ". "),
			("str", ",", ", "),
			("re", r"([0-9]+)\s*,\s*([0-9]+)", r"\g<1>,\g<2>"),
			("str", "!", "! "),
			("str", "?", "? "),

			# normalize whitespace before punctuation symbols.
			("re", r"\s+\:", ":"),
			("re", r"\s+\;", ";"),
			("re", r"\s+\.", "."),
			("re", r"\s+\,", ","),
			("re", r"\s+\!", "!"),
			("re", r"\s+\?", "?"),

			# normalize whitespace before and after parentheses.
			("re", r"\s+\)", ")"),
			("re", r"\(\s+", "("),

			("re", r"\)\s+\:", "):"),
			("re", r"\)\s+\;", ");"),
			("re", r"\)\s+\.", ")."),
			("re", r"\)\s+\,", "),"),
			("re", r"\)\s+\!", ")!"),
			("re", r"\)\s+\?", ")?"),

			("re", r"\.\s+\)", ".)"),
			("re", r"\!\s+\)", "!)"),
			("re", r"\?\s+\)", "?)"),

			# normalize whitespace around dashes.
			("re", r"\-([^\s-])", r"- \g<1>"),
			("re", r"([^\s-])\-", r"\g<1> -"),
			("re", r"([0-9]+)\s*\-\-\s*([0-9]+)", r"\g<1>--\g<2>"),

            ("re", r"([0-9]+)\s*--\s*([0-9]+)", r"\g<1>--\g<2>"),
            ("re", r"([0-9]+)\s*-\s*([0-9]+)", r"\g<1>-\g<2>"),

			# normalize whitespace before percentage sign.
            ("re", r"([0-9]+)\s+%", r"\g<1>%"),

			# normalize whitespace in lists of numbers.
            ("re", r"([0-9]+)\s*,\s*([0-9]+)", r"\g<1>,\g<2>"),

            # normalize multiple whitespace symbols.
			("re", r"\s+", " ")
		]
	}
}
