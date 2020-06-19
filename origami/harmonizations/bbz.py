# BBZ-specific transcription harmonization schema, built for use
# with text from the Berliner Börsen-Zeitung.

{
	"channels": {
		"unstyled": {
			"transform": "unstyled",
			"alphabet": {
				"lowercase": "abcdefghijklmnopqrstuvwxyzß",
				"uppercase": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",

				"lowercase_diacritic": "äöüàáâôéèêëç",
				"uppercase_diacritic": "ÄÖÜ",

				"punctuation": "-?!.,:; ",
				"quotes": "‚'",
				"brackets": "()<>",
				"slashes": "/",
				"math": "+=%",

				"footnote": "*†",

				"digits": "1234567890",
				"currencies": "£$",

				"symbols": "§&△"
			},
			"tests": ["common", "unstyled"]
		},
		"styled": {
			"transform": "styled",
			"alphabet": {
				"lowercase": "abcdefghijklmnopqrstuvwxyzß",
				"uppercase": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",

				"lowercase_diacritic": "äöüàáâôéèêëç",
				"uppercase_diacritic": "ÄÖÜ",

				"punctutation": "-?!.,:;‚' ",

				"brackets": "()<>",
				"slashes": "/",
				"math": "+=%",

				"footnote": "*†",

				"digits": "1234567890",
				"currencies": "£$",

				"symbols": "§&△",
				"styles": "{}[]"
			},
			"tests": ["common", "styled"]
		}
	},
	"tests": {
		"styled": [
			("{a} [b]", "{a} [b]"),
			("- [a]", "- [a]"),
			("[- a]", "- [a]"),
			("[-a]", "-[a]"),
			("-[a]", "-[a]"),
			("--[a]", "--[a]"),

			("[a.]", "[a]."),
			("[a,]", "[a],"),
			("[a:]", "[a]:"),
			("[a;]", "[a];"),
			("[a?]", "[a]?"),
			("[a!]", "[a]!"),

			("[a) {b}]", "[a) {b}]")
		],
		"unstyled": [
			("{a} [b]", "a b"),
		],
		"common": [
			("a  b c", "a b c"),
			("a.b", "a. b"),
			("3.4", "3.4"),
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
			("12 -- 34", "12 -- 34"),
			("Stückà3", "Stück à 3"),
			("a\"b", "a'' b"),
			("a'''b", "a''' b"),
			("a„b", "a ‚‚b"),
			("a‚‚‚b", "a ‚‚‚b"),
			("3 + 4", "3 + 4"),
			("a + 4", "a + 4"),
			("a - b", "a - b"),
			("a- b", "a- b"),
			("a-", "a-"),
			("a,b", "a, b"),
			("3,4", "3,4"),
			("3, 4", "3, 4"),
			("a.b", "a. b"),
			("a.)", "a.)"),
			("a!b", "a! b"),
			("a!)", "a!)"),
			("a?b", "a? b"),
			("a?)", "a?)"),
			("Thlr. .", "Thlr.."),
			("Thlr..", "Thlr..")
		]
	},
	"transforms": {
		"unstyled": [
			("re", r"{|}|\[|\]", ""),
			("tfm", "default")
		],
		"styled": [
			("tfm", "default"),
			("re", r"([^\w]+)\]", r"]\g<1>"),
			("re", r"\[([^\w]+)", r"\g<1>["),
			("re", r"([^\w]+)\}", r"}\g<1>"),
			("re", r"\{([^\w]+)", r"\g<1>{"),
			("re", r"\s+", " ")
		],
		"default": [
			# normalize style annotations.
			("re", r"\{\s*\[", "[{"),
			("re", r"\]\s*\}", "}]"),

			# normalize dashes.
			("str", "―", "--"),
			("str", "•", "-"),

			# normalize quotes and apostrophes.
            ("str", "”", "''"),
            ("str", "„", "‚‚"),
			("str", "\"", "''"),

			# normalize whitespace before and after quotes.
			("re", r"([^‚\s])‚‚", r"\g<1> ‚‚"),
			("re", r"‚‚\s+", "‚‚"),

			("re", r"''([^'\s])", r"'' \g<1>"),
			("re", r"\s+''", "''"),

			# expand fractions and other composite symbols.
			("str", "½", "<1/2>"),
			("str", "¼", "<1/4>"),
			("str", "¾", "<3/4>"),
			("str", "°", "<0 "),

			# normalize whitespace around operator symbols.
			#("re", r"([^0-9]+)\+([0-9]+)", r"\g<1> \+ \g<2>"),
			("re", r"à([0-9]+)", r" à \g<1>"),

			# normalize whitespace after punctuation symbols.
			("str", ":", ": "),
			("str", ";", "; "),
			("re", r"\.\s*([^\W\d]+)", r". \g<1>"),
			("re", r"([^0-9]+)\s*,\s*([^0-9]+)", r"\g<1>, \g<2>"),
			("re", r"\!\s*([^\W\d]+)", r"! \g<1>"),
			("re", r"\?\s*([^\W\d]+)", r"? \g<1>"),
			("re", r"Thlr\.\s+\.", "Thlr.."),

			# normalize number formatting.
			#("re", r"([0-9]+)\s*,\s*([0-9]+)", r"\g<1>,\g<2>"),

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
			#("re", r"\-([^\s-])", r"- \g<1>"),
			#("re", r"([^\s-])\-", r"\g<1> -"),

            ("re", r"([0-9]+)\s*--\s*([0-9]+)", r"\g<1> -- \g<2>"),
            #("re", r"([0-9]+)\s*-\s*([0-9]+)", r"\g<1>-\g<2>"),

			# normalize percentage formatting.
			("re", r"([0-9]+)\s+%", r"\g<1>%"),

			# normalize multiple whitespace symbols.
			("re", r"\s+", " ")
		]
	}
}
