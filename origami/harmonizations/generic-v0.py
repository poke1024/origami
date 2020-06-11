{
	"channels": {
		"default": {
			"transform": "default",
			"alphabet": "_-?!.,:;‚'äöüÄÖÜßàáâôéè1234567890%§&£*+=/() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ«»çꝛ⏓⏑òøåæûëñêÅÊî<>Ἰϰχεάù[]Æ|€",
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

			("str", "⸗", "-"),  # in Calamari fraktur 2020 prediction
			("str", "·", "-"),  # in gt4hist prediction
			("str", "]", ")"),  # in gt4hist prediction
			("str", "ẽ", "e"),  # in gt4hist prediction
			("str", "ů", "u"),  # in gt4hist prediction
			("str", "ã", "a"),  # in gt4hist prediction
			("str", "ũ", "u"),  # in gt4hist prediction
			("str", "õ", "o"),  # in gt4hist prediction
			("str", "Ʒ", "3"),  # in gt4hist prediction
			("str", "ʒ", "3"),  # in gt4hist prediction
			("str", "ꝗ", "q"),  # in gt4hist prediction
			("str", "ꝓ", "p"),  # in gt4hist prediction
			("str", "ĩ", "i"),  # in gt4hist prediction
			("str", "ð", "d"),  # in gt4hist prediction
			("str", "¶", ""),  # in tesseract prediction
			("str", "…", "..."), # in tesseract prediction
			("str", ("chr", 0x303), ""),  # in tesseract prediction
			("str", "ꝑ", "p"),  # in tesseract prediction

			("str", "ſ", "s"),
			("str", "¬", "-"),  # convert double-dash to short dash. (NZZ)
			("str", "—", "―"),  # convert NZZ long dash to short dash. (NZZ)
			("str", "–", "-"),  # 1832-lenau_gedichte_00587

			("str", ("chr", 868), "e"),  # upper e
			("str", ("chr", 0x2018), "'"),  # apostrophe
			("str", "’", "'"),
            ("str", "\"", "''"),

			("str", "½", " 1/2 "),
			("str", "¼", " 1/4 "),
			("str", "¾", " 3/4 "),
			("str", "⅔", " 2/3 "),
			("str", "―", "--"),

			("str", ":", ": "),
			("str", ";", "; "),
			("str", ".", ". "),
			#("re", r"([0-9]+)\s*\.\s*([0-9]+)", r"\g<1>.\g<2>"),
			("str", ",", ", "),
			("re", r"([0-9]+)\s*,\s*([0-9]+)", r"\g<1>,\g<2>"),
			("str", "!", "! "),
			("str", "?", "? "),

			("str", "„", " „"),
			("str", "”", "” "),

			("re", r"\s+\:", ":"),
			("re", r"\s+\;", ";"),
			("re", r"\s+\.", "."),
			("re", r"\s+\,", ","),
			("re", r"\s+\!", "!"),
			("re", r"\s+\?", "?"),

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

			("re", r"„\s+", "„"),
			("re", r"\s+”", "”"),

			("re", r"\-([^\s-])", r"- \g<1>"),
			("re", r"([^\s-])\-", r"\g<1> -"),
			("re", r"([0-9]+)\s*\-\-\s*([0-9]+)", r"\g<1>--\g<2>"),

            ("re", r"([0-9]+)\s+%", r"\g<1>%"),
            ("re", r"([0-9]+)\s*--\s*([0-9]+)", r"\g<1>--\g<2>"),
            ("re", r"([0-9]+)\s*-\s*([0-9]+)", r"\g<1>-\g<2>"),
            ("re", r"([0-9]+)\s*,\s*([0-9]+)", r"\g<1>,\g<2>"),

            ("str", "”", "''"),
            ("str", "„", "‚‚"),
            ("str", "“", "''"),
			("re", r"\s+", " ")
		]
	}
}
