import functools


class FunctionProxy:
	def __init__(self):
		self.kwargs = dict()

	def __call__(self, **kwargs):
		self.kwargs = kwargs
		return self


def build_func_from_string(spec, funcs0):
	names = list(funcs0.keys())

	locals = dict((x, FunctionProxy()) for x in names)

	funcs = dict()
	for x in names:
		funcs[id(locals[x])] = funcs0[x]

	data = eval(spec, locals)

	if not isinstance(data, FunctionProxy):
		raise ValueError(data)

	return functools.partial(funcs[id(data)], **data.kwargs)
