# -*- coding: utf-8 -*-
# noqa: D205,D400
"""
Indicators utilities
====================

The `Indicator` class wraps indices computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the output object.

There are many ways to construct indicators. A good place to start is `this notebook <notebooks/extendxclim.ipynb#Defining-new-indicators>`_.

Dictionary and YAML parser
--------------------------

To construct indicators dynamically, xclim can also use dictionaries and parse them from YAML files.
This is especially useful for generating whole indicator "submodules" from files.
This functionality is based on and extends the work of [clix-meta](https://github.com/clix-meta/clix-meta/).

YAML file structure
~~~~~~~~~~~~~~~~~~~

Indicator-defining yaml files are structured in the following way:

.. code-block:: yaml

    module: <module name>  # Defaults to the file name
    realm: <realm>  # If given here, applies to all indicators that do no give it.
    base: <base indicator class>  # Defaults to "Daily"
    doc: <module docstring>  # Defaults to a minimal header, only valid if the module doesn't already exists.
    indices:
      <identifier>:
        base: <base indicator class>  # Defaults to module-wide base class or "Daily".
        realm: <realm>  # Defaults to the module-wide realm or "atmos"
        reference: <references>
        references: <references>  # Plural or singular accepted (for harmonizing clix-meta and xclim)
        keywords: <keywords>
        notes: <notes>
        title: <title>
        abstract: <abstract>
        period:  # If given, both "allowed" and "default" must also be given.
          allowed:  # A list of allowed periods (resampling frequencies)
            annual:  # Translates to "A" (includes "Y")
            seasonal:  # Translates to "Q"
            monthly:  # Translates to "M"
            weekly:  # Translates to "W"
          default: annual  #  Translates to "YS", "QS-DEC", "MS" or "W-SUN". See xclim.core.units.FREQ_NAMES.
        output:
          var_name: <var_name>  # Defaults to "identifier",
          standard_name: <standard_name>
          long_name: <long_name>
          description: <description>
          comment: <comment>
          units: <units>  # Defaults to ""
          cell_methods:
            - <dim1> : <method 1>
            ...

        index_function:
          name: <function name>  # Refering to a function in xclim.indices.generic or xclim.indices
          parameters:  # See below for details on that section.
            <param name>  # Refering to a parameter of the function above.
              kind: <param kind>  # Optional, one of quantity, operator or reducer
              data: <param data>
              units: <param units>
              operator: <param data>
              reducer: <param data>
            ...

        input:
          <var1> : <variable type 1>  # <var1> refers to a name in the function above, see below.
          ...
      ...  # and so on.

All fields are optional. Other fields can be found in the yaml file, but they will not be used by xclim.
In the following, the section under `<identifier>` is refered to as `data`. When creating indicators from
a dictionary, with :py:meth:`Indicator.from_dict`, the input dict must follow the structure of `data`.

Indicator parameters
~~~~~~~~~~~~~~~~~~~~
`clix-meta` defines three kinds of parameters:

    - "quantity", a quantity with a magnitude and some units, (equivalent to xclim.core.utils.InputKind.QUANTITY_STR)
      The value is given through the magnitude in "data" and units in "units".
    - "operator", one of "<", "<=", ">", ">=", "==", "!=", an operator for conditional computations.
      The value is given in "operator".
    - "reducer", one of "maximum", "minimum", "mean", "sum", a reducing method name.
      The value is given in "reducer".

xclim supports both this syntax and a simpler one where only the "data" key is given.
As YAML is able to cast simple python literals, no passing of "kind" is needed, if a string parameter could be
mistranslated to a boolean or a number, simply use quotes to isolate it. To pass a number sequence, use
the yaml list syntax.

Inputs
~~~~~~
As xclim has strict definitions of possible input variables (see :py:data:`xclim.core.yaml.variables`),
the mapping of `data.input` simply links a variable name from the function in `data.index_function.name`
to one of those official variables.

"""
import logging
import re
import warnings
import weakref
from collections import OrderedDict, defaultdict
from copy import deepcopy
from inspect import Parameter, _empty, signature
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
from boltons.funcutils import copy_function, wraps
from xarray import DataArray, Dataset
from yaml import safe_load

from .. import indices
from . import datachecks
from .calendar import parse_offset
from .cfchecks import generate_cfcheck
from .formatting import (
    AttrFormatter,
    default_formatter,
    generate_indicator_docstring,
    merge_attributes,
    parse_cell_methods,
    parse_doc,
    update_history,
)
from .locales import TRANSLATABLE_ATTRS, get_local_attrs, get_local_formatter
from .options import MISSING_METHODS, MISSING_OPTIONS, OPTIONS
from .units import FREQ_NAMES, convert_units_to, declare_units, units
from .utils import (
    MissingVariableError,
    infer_kind_from_parameter,
    variables,
    wrapped_partial,
)

# Indicators registry
registry = dict()  # Main class registry
_indicators_registry = defaultdict(list)  # Private instance registry


class IndicatorRegistrar:
    """Climate Indicator registering object."""

    def __new__(cls):
        """Add subclass to registry."""
        name = cls.__name__.upper()
        module = cls.__module__
        # If the module is not one of xclim's default, prepend the submodule name.
        if module.startswith("xclim.indicators"):
            submodule = module.split(".")[2]
            if submodule not in ["atmos", "land", "ocean", "seaIce"]:
                name = f"{submodule}.{name}"
        else:
            name = f"{module}.{name}"
        if name in registry:
            warnings.warn(
                f"Class {name} already exists and will be overwritten.", stacklevel=1
            )
        registry[name] = cls
        cls._registry_id = name
        return super().__new__(cls)

    def __init__(self):
        _indicators_registry[self.__class__].append(weakref.ref(self))

    @classmethod
    def get_instance(cls):
        """Return first found instance.

        Raises `ValueError` if no instance exists.
        """
        for inst_ref in _indicators_registry[cls]:
            inst = inst_ref()
            if inst is not None:
                return inst
        raise ValueError(
            f"There is no existing instance of {cls.__name__}. Either none were created or they were all garbage-collected."
        )


class Indicator(IndicatorRegistrar):
    r"""Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. Some of these attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also creates and registers a custom subclass.

    Parameters in `Indicator._cf_names` will be added to the output variable(s). When creating new `Indicators` subclasses,
    if the compute function returns multiple variables, attributes may be given as lists of strings or strings.
    In the latter case, the same value is used on all variables.

    Compared to their base `compute` function, indicators add the possibility of using dataset as input,
    with the injected argument `ds` in the call signature. All arguments that were indicated by the compute function
    to be DataArrays through annotations will be promoted to also accept strings that correspond to variable names
    in the `ds` dataset.

    Parameters
    ----------
    identifier: str
      Unique ID for class registry, should be a valid slug.
    realm : {'atmos', 'seaIce', 'land', 'ocean'}
      General domain of validity of the indicator. Indicators created outside xclim.indicators must set this attribute.
    compute: func
      The function computing the indicators. It should return one or more DataArray.
    var_name: str or Sequence[str]
      Output variable(s) name(s). May use tags {<tag>}. If the indicator outputs multiple variables,
      var_name *must* be a list of the same length.
    standard_name: str or Sequence[str]
      Variable name (CF).
    long_name: str or Sequence[str]
      Descriptive variable name. Parsed from `compute` docstring if not given.
    units: str or Sequence[str]
      Representative units of the physical quantity (CF).
    cell_methods: str or Sequence[str]
      List of blank-separated words of the form "name: method" (CF).
    description: str or Sequence[str]
      Sentence meant to clarify the qualifiers of the fundamental quantities, such as which
      surface a quantity is defined on or what the flux sign conventions are.
    comment: str or Sequence[str]
      Miscellaneous information about the data or methods used to produce it.
    title: str
      A succinct description of what is in the computed outputs. Parsed from `compute` docstring if None.
    abstract: str
      A long description of what is in the computed outputs. Parsed from `compute` docstring if None.
    keywords: str
      Comma separated list of keywords. Parsed from `compute` docstring if None.
    references: str
      Published or web-based references that describe the data or methods used to produce it. Parsed from
      `compute` docstring if None.
    notes: str
      Notes regarding computing function, for example the mathematical formulation. Parsed from `compute`
      docstring if None.
    missing: {any, wmo, pct, at_least_n, skip, from_context}
      The name of the missing value method. See `xclim.core.missing.MissingBase` to create new custom methods. If
      None, this will be determined by the global configuration (see `xclim.set_options`). Defaults to "from_context".
    freq: {"D", "H", None}
      The expected frequency of the input data. Use None if irrelevant.
    missing_options : dict, None
      Arguments to pass to the `missing` function. If None, this will be determined by the global configuration.
    context: str
      The `pint` unit context, for example use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.
    allowed_periods : Sequence[str], optional
      A list of allowed periods, i.e. base parts of the `freq` parameter. For example, indicators meant to be
      computed annually only will have `allowed_periods=["Y", "A"]`. `None` means, "any period" or that the
      indicator doesn't take a `freq` argument.

    Notes
    -----
    All subclasses created are available in the `registry` attribute and can be used to define custom subclasses
    or parse all available instances.

    """

    #: Number of input DataArray variables. Should be updated by subclasses if needed.
    #: This number sets which inputs are passed to the tests.
    nvar = 1

    # Allowed metadata attributes on the output variables
    _cf_names = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
    ]

    # metadata fields that are formatted as free text.
    _text_fields = ["long_name", "description", "comment"]

    _funcs = ["compute", "cfcheck", "datacheck"]

    # Will become the class's name
    identifier = None

    missing = "from_context"
    missing_options = None
    context = "none"
    freq = None
    allowed_periods = None

    # Variable metadata (_cf_names, those that can be lists or strings)
    # A developper should access those through cf_attrs on instances
    var_name = None
    standard_name = ""
    long_name = ""
    units = ""
    cell_methods = ""
    description = ""
    comment = ""

    # Global metadata (must be strings, not attributed to the output)
    realm = None
    title = ""
    abstract = ""
    keywords = ""
    references = ""
    notes = ""

    parameters: Mapping[str, Any]
    """A dictionary mapping metadata about the input parameters to the indicator.

       Contains : "default", "description", "kind" and, sometimes, "units" and "choices".
       "kind" refers to the constants of :py:class:`xclim.core.utils.InputKind`.
    """

    cf_attrs: Sequence[Mapping[str, Any]]
    """A list of metadata information for each output of the indicator.

       It minimally contains a "var_name" entry, and may contain : "standard_name", "long_name",
       "units", "cell_methods", "description" and "comment".
    """

    def __new__(cls, **kwds):
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", cls.identifier)
        if identifier is None:
            raise AttributeError("`identifier` has not been set.")

        kwds["var_name"] = kwds.get("var_name", cls.var_name) or identifier

        # Parse and update compute's signature.
        kwds["compute"] = kwds.get("compute", None) or cls.compute
        # Updated to allow string variable names and the ds arg.
        # Parse docstring of the compute function, its signature and its parameters
        kwds["_indcompute"], docmeta, params = _parse_indice(
            kwds["compute"],
            passed=kwds.get("parameters"),
            ds={
                "annotation": Dataset,
                "description": "A dataset with the variables given by name.",
            },
        )

        # The update signature
        kwds["_sig"] = kwds["_indcompute"].__signature__
        # The input parameters' name
        kwds["_parameters"] = tuple(kwds["_sig"].parameters.keys())

        # All fields parsed by parse_doc except "parameters"
        # i.e. : title, abstract, notes, references, long_name
        for name, value in docmeta.items():
            if not getattr(cls, name):
                # Set if neither the class attr is set nor the kwds attr
                kwds.setdefault(name, value)

        # The input parameters' metadata
        # We dump whatever the base class had and take what was parsed from the current compute function.
        kwds["parameters"] = params

        # Parse kwds to organize cf_attrs
        # Must be done after parsing var_name
        # And before converting callables to staticmethods
        kwds["cf_attrs"] = cls._parse_cf_attrs(kwds)

        # Convert function objects to static methods.
        for key in cls._funcs + cls._cf_names:
            if key in kwds and callable(kwds[key]):
                kwds[key] = staticmethod(kwds[key])

        # Infer realm for built-in xclim instances
        if cls.__module__.startswith(__package__.split(".")[0]):
            xclim_realm = cls.__module__.split(".")[2]
        else:
            xclim_realm = None
        # Priority given to passed realm -> parent's realm -> location of the class declaration (official inds only)
        kwds.setdefault("realm", cls.realm or xclim_realm)
        if kwds["realm"] not in ["atmos", "seaIce", "land", "ocean"]:
            raise AttributeError(
                "Indicator's realm must be given as one of 'atmos', 'seaIce', 'land' or 'ocean'"
            )

        # Create new class object
        new = type(identifier.upper(), (cls,), kwds)

        # Forcing the module is there so YAML-generated submodules are correctly seen by IndicatorRegistrar.
        if "module" in kwds:
            new.__module__ = f"xclim.indicators.{kwds['module']}"
        else:
            # If the module was not forced, set the module to the base class' module.
            # Otherwise all indicators will have module `xclim.core.indicator`.
            new.__module__ = cls.__module__

        # Generate docstring
        new._indcompute.__doc__ = new.__doc__ = generate_indicator_docstring(new)

        #  Add the created class to the registry
        # This will create an instance from the new class and call __init__.
        return super().__new__(new)

    @classmethod
    def _parse_cf_attrs(
        cls, kwds: Dict[str, Any]
    ) -> Union[List[Dict[str, str]], List[Dict[str, Union[str, Callable]]]]:
        """CF-compliant metadata attributes for all output variables."""
        # Get number of outputs
        n_outs = (
            len(kwds["var_name"]) if isinstance(kwds["var_name"], (list, tuple)) else 1
        )

        # Populate cf_attrs from attribute set during class creation and __new__
        cf_attrs = [{} for i in range(n_outs)]
        for name in cls._cf_names:
            values = kwds.get(name, getattr(cls, name))
            if not isinstance(values, (list, tuple)):
                values = [values] * n_outs
            elif len(values) != n_outs:
                raise ValueError(
                    f"Attribute {name} has {len(values)} elements but should have {n_outs} according to passed var_name."
                )
            for attrs, value in zip(cf_attrs, values):
                if value:
                    attrs[name] = value
        return cf_attrs

    @classmethod
    def from_dict(
        cls,
        data: dict,
        identifier: str,
        module: Optional[str] = None,
        realm: Optional[str] = None,
        keywords: Optional[str] = None,
        references: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Create an indicator subclass and instance from a dictionary of parameters.

        Parameters
        ----------
        data: dict
          The exact structure of this dictionary is detailed in the submodule documentation.
        identifier : str
          The name of the subclass and internal indicator name.
        module : str
          The module name of the indicator. This is meant to be used only if the indicator
          is part of a dynamically generated submodule, to override the module of the base class.
        realm: str, optional
        keywords: str, optional
        references str, optional
        notes: str, optional
          Other indicator attributes to fill in for missing values in the individual definition.
        """
        # Make cell methods. YAML will generate a list-of-dict structure, put it back in a space-divided string
        if data.get("output", {}).get("cell_methods") is not None:
            cell_methods = parse_cell_methods(data["output"]["cell_methods"])
        else:
            cell_methods = None

        params = {}
        if "input" in data:
            # Override input metadata
            input_units = {}
            nvar = len(data["input"])
            for varname, name in data["input"].items():
                # Indicator's new will put the name of the variable as its default,
                # we override this with the real variable name.
                # Also take the dimensionaliy and description from the yaml of official variables.
                # Description overrides the one parsed from the generic compute docstring
                # Dimensionality goes into the declare_units wrapper.
                params[varname] = {
                    "default": name,
                    "description": variables[name]["description"],
                }
                input_units[varname] = variables[name]["dimensionality"]

            cfcheck = generate_cfcheck(*[varname for varname in data["input"].values()])
        else:
            nvar = None
            cfcheck = None
            input_units = None

        metadata_placeholders = {}
        if "index_function" in data:
            # Generate compute function
            # data.index_function.name refers to a function in xclim.indices.generic or xclim.indices (in this order of priority).
            # data.index_function.parameters is a list of injected arguments.
            funcname = data["index_function"].get("name")
            if funcname is None:
                # No index function given, reuse the one from the base class.
                compute = cls.compute
            else:
                compute = getattr(
                    indices.generic, funcname, getattr(indices, funcname, None)
                )
                if compute is None:
                    raise ImportError(
                        f"Indice function {funcname} not found in xclim.indices or xclim.indices.generic."
                    )

            injected_params = {}
            # In clix-meta, when there are no parameters, the key is still there with a None value.
            for name, param in (data["index_function"].get("parameters") or {}).items():
                # Handle clix-meta cases
                if param.get("kind") == "quantity" and isinstance(
                    param["data"], (str, int, float)
                ):
                    # A string with units, but not a placeholder (where data is a dict)
                    value = f"{param['data']} {param['units']}"
                elif param.get("kind") in ["reducer", "operator"]:
                    # clix-meta defined kinds :value is stored in a field of the same name as the kind.
                    value = param[param["kind"]]
                else:
                    # All other xclim-defined kinds in "data"
                    value = param["data"]

                if isinstance(value, dict):
                    # User-chosen parameter. placeholder.
                    # It should be a string, this is a bug from clix-meta.
                    value = list(value.keys())[0]
                    params[name] = {
                        "default": param.get("default"),
                        "description": param.get(
                            "description", param.get("standard_name", name)
                        ),
                    }
                    if "units" in param:
                        params[name]["units"] = param["units"]
                        input_units = input_units or {}
                        input_units[name] = param["units"]
                    # We will need to replace placeholders in metadata strings (only for clix-meta indicators)
                    if value != name:
                        metadata_placeholders["{" + value + "}"] = "{" + name + "}"
                else:
                    # Injected parameter
                    injected_params[name] = value

            if input_units is not None:
                compute = declare_units(**input_units)(compute)

            compute = wrapped_partial(compute, **injected_params)
        else:
            compute = None

        # Allowed resampling frequencies
        if "period" in data:
            params["freq"] = {"default": FREQ_NAMES[data["period"]["default"]][1]}
            allowed_periods = []
            for period_name in data["period"]["allowed"]:
                allowed_periods.append(FREQ_NAMES[period_name][0])
        else:
            allowed_periods = None

        kwargs = dict(
            # General
            identifier=identifier,
            module=module,
            realm=data.get("realm", realm),
            keywords=data.get("keywords", keywords),
            references=data.get("references", data.get("reference", references)),
            notes=data.get("notes", notes),
            # Indicator-specific metadata
            title=data.get("title"),
            abstract=data.get("abstract"),
            # Output meta
            var_name=data.get("output", {}).get("var_name", identifier),
            standard_name=data.get("output", {}).get("standard_name"),
            long_name=data.get("output", {}).get("long_name"),
            description=data.get("output", {}).get("description"),
            comment=data.get("output", {}).get("comment"),
            units=data.get("output", {}).get("units"),
            cell_methods=cell_methods,
            # Input data, override defaults given in generic compute's signature.
            parameters=params or None,  # None if an empty dict
            nvar=nvar,
            compute=compute,
            # Checks
            cfcheck=cfcheck,
            allowed_periods=allowed_periods,
        )

        for cf_name in cls._cf_names:
            if isinstance(kwargs[cf_name], str):
                for old, new in metadata_placeholders.items():
                    kwargs[cf_name] = kwargs[cf_name].replace(old, new)

        # Remove kwargs passed as "None", they will be taken from the base class instead.
        # For most parameters it would be ok to pass a None anyway (we figure that out in __new__),
        # but some (like nvar) would not like that.
        return cls(**{k: v for k, v in kwargs.items() if v is not None})

    def __init__(self, **kwds):
        """Run checks and organizes the metadata."""
        # keywords of kwds that are class attributes have already been set in __new__
        self._check_identifier(self.identifier)
        if self.missing == "from_context" and self.missing_options is not None:
            raise ValueError(
                "Cannot set `missing_options` with `missing` method being from context."
            )

        # Validate hard-coded missing options
        kls = MISSING_METHODS[self.missing]
        self._missing = kls.execute
        if self.missing_options:
            kls.validate(**self.missing_options)

        # Validation is done : register the instance.
        super().__init__()

        # Update call signature
        self.__call__ = wraps(self._indcompute)(self.__call__)

    def __call__(self, *args, **kwds):
        """Call function of Indicator class."""
        # For convenience
        n_outs = len(self.cf_attrs)

        # Bind call arguments to `compute` arguments and set defaults.
        ba = self._sig.bind(*args, **kwds)
        ba.apply_defaults()

        # Assign inputs passed as strings from ds.
        self._assign_named_args(ba)

        # Assume the first arguments are always the DataArrays.
        # Only the first nvar inputs are checked (data + cf checks)
        das = OrderedDict()
        for name in self._parameters[: self.nvar]:
            das[name] = ba.arguments.pop(name)

        # Metadata attributes from templates
        var_id = None
        var_attrs = []
        for attrs in self.cf_attrs:
            if n_outs > 1:
                var_id = f"{self._registry_id}.{attrs['var_name']}"
            var_attrs.append(
                self._update_attrs(ba, das, attrs, names=self._cf_names, var_id=var_id)
            )

        # Pre-computation validation checks on DataArray arguments
        self._bind_call(self.datacheck, **das)
        self._bind_call(self.cfcheck, **das)

        # Check if the period is allowed:
        if (
            self.allowed_periods is not None
            and "freq" in kwds
            and parse_offset(kwds["freq"])[1] not in self.allowed_periods
        ):
            raise ValueError(
                f"Resampling frequency {kwds['freq']} is not allowed for indicator {self.identifier} (needs something equivalent to one of {self.allowed_periods})."
            )

        # Compute the indicator values, ignoring NaNs and missing values.
        outs = self.compute(**das, **ba.kwargs)
        if isinstance(outs, DataArray):
            outs = [outs]
        if len(outs) != n_outs:
            raise ValueError(
                f"Indicator {self.identifier} was wrongly defined. Expected {n_outs} outputs, got {len(outs)}."
            )

        # Convert to output units
        outs = [
            convert_units_to(out, attrs.get("units", ""), self.context)
            for out, attrs in zip(outs, var_attrs)
        ]

        # Update variable attributes
        for out, attrs in zip(outs, var_attrs):
            var_name = attrs.pop("var_name")
            out.attrs.update(attrs)
            out.name = var_name

        # Mask results that do not meet criteria defined by the `missing` method.
        # This means all variables must have the same dimensions...
        mask = self._mask(*das.values(), **ba.arguments)
        outs = [out.where(~mask) for out in outs]

        # Return a single DataArray in case of single output, otherwise a tuple
        if n_outs == 1:
            return outs[0]
        return tuple(outs)

    def _assign_named_args(self, ba):
        """Assign inputs passed as strings from ds."""
        ds = ba.arguments.pop("ds")
        for name, param in self._sig.parameters.items():
            if param.annotation is Union[str, DataArray] and isinstance(
                ba.arguments[name], str
            ):
                if ds is not None:
                    try:
                        ba.arguments[name] = ds[ba.arguments[name]]
                    except KeyError:
                        raise MissingVariableError(
                            f"For input '{name}', variable '{ba.arguments[name]}' was not found in the input dataset."
                        )
                else:
                    raise ValueError(
                        f"Passing variable names as string requires giving the `ds` dataset (got {name}='{ba.arguments[name]}')"
                    )

    def _bind_call(self, func, **das):
        """Call function using `__call__` `DataArray` arguments.

        This will try to bind keyword arguments to `func` arguments. If this fails, `func` is called with positional
        arguments only.

        Notes
        -----
        This method is used to support two main use cases.

        In use case #1, we have two compute functions with arguments in a different order:
            `func1(tasmin, tasmax)` and `func2(tasmax, tasmin)`

        In use case #2, we have two compute functions with arguments that have different names:
            `generic_func(da)` and `custom_func(tas)`

        For each case, we want to define a single `cfcheck` and `datacheck` methods that will work with both compute
        functions.

        Passing a dictionary of arguments will solve #1, but not #2.
        """
        # First try to bind arguments to function.
        try:
            ba = signature(func).bind(**das)
        except TypeError:
            # If this fails, simply call the function using positional arguments
            return func(*das.values())
        else:
            # Call the func using bound arguments
            return func(*ba.args, **ba.kwargs)

    @classmethod
    def _update_attrs(cls, ba, das, attrs, var_id=None, names=None):
        """Format attributes with the run-time values of `compute` call parameters.

        Cell methods and xclim_history attributes are updated, adding to existing values. The language of the string is
        taken from the `OPTIONS` configuration dictionary.

        Parameters
        ----------
        das: tuple
          Input arrays.
        ba: bound argument object
          Keyword arguments of the `compute` call.
        attrs : Mapping[str, str]
          The attributes to format and update.
        var_id : str
          The identifier to use when requesting the attributes translations.
          Defaults to the class name (for the translations) or the `identifier` field of the class (for the xclim_history attribute).
          If given, the identifier will be converted to uppercase to get the translation attributes.
          This is meant for multi-outputs indicators.
        names : Sequence[str]
          List of attribute names for which to get a translation.

        Returns
        -------
        dict
          Attributes with {} expressions replaced by call argument values. With updated `cell_methods` and `xclim_history`.
          `cell_methods` is not added is `names` is given and those not contain `cell_methods`.
        """
        args = ba.arguments
        out = cls._format(attrs, args)
        for locale in OPTIONS["metadata_locales"]:
            out.update(
                cls._format(
                    get_local_attrs(
                        var_id or cls._registry_id,
                        locale,
                        names=names or list(attrs.keys()),
                        append_locale_name=True,
                    ),
                    args=args,
                    formatter=get_local_formatter(locale),
                )
            )

        # Generate a signature string for the history attribute
        # We remove annotations, replace default float/int/str by values
        # and replace others by type
        callstr = []
        for (k, v) in das.items():
            callstr.append(f"{k}=<array>")
        for (k, v) in ba.arguments.items():
            if isinstance(v, (float, int, str)):
                callstr.append(f"{k}={v!r}")  # repr so strings have ' '
            else:
                callstr.append(
                    f"{k}={type(v)}"
                )  # don't take chance of having unprintable values

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        if names is None or "cell_methods" in names:
            attrs["cell_methods"] = merge_attributes(
                "cell_methods", new_line=" ", missing_str=None, **das
            )
            if "cell_methods" in out:
                attrs["cell_methods"] += " " + out.pop("cell_methods")

        attrs["xclim_history"] = update_history(
            f"{var_id or cls._registry_id}({', '.join(callstr)})",
            new_name=out.get("var_name"),
            **das,
        )

        attrs.update(out)
        return attrs

    @staticmethod
    def _check_identifier(identifier: str) -> None:
        """Verify that the identifier is a proper slug."""
        if not re.match(r"^[-\w]+$", identifier):
            warnings.warn(
                "The identifier contains non-alphanumeric characters. It could make life "
                "difficult for downstream software reusing this class.",
                UserWarning,
            )

    def translate_attrs(
        self, locale: Union[str, Sequence[str]], fill_missing: bool = True
    ):
        """Return a dictionary of unformated translated translatable attributes.

        Translatable attributes are defined in :py:const:`xclim.core.locales.TRANSLATABLE_ATTRS`.

        Parameters
        ----------
        locale : Union[str, Sequence[str]]
            The POSIX name of the locale or a tuple of a locale name and a path to a
            json file defining the translations. See `xclim.locale` for details.
        fill_missing : bool
            If True (default fill the missing attributes by their english values.
        """

        def _translate(var_id, var_attrs, names):
            attrs = get_local_attrs(
                var_id,
                locale,
                names=names,
                append_locale_name=False,
            )
            if fill_missing:
                for name in names:
                    if name not in attrs and var_attrs.get(name):
                        attrs[name] = var_attrs.get(name)
            return attrs

        # Translate global attrs
        attrid = self._registry_id
        attrs = _translate(
            attrid,
            self.__dict__,
            # Translate only translatable attrs that are not variable attrs
            set(TRANSLATABLE_ATTRS).difference(set(self._cf_names)),
        )
        # Translate variable attrs
        attrs["outputs"] = []
        for var_attrs in self.cf_attrs:  # Translate for each variable
            if len(self.cf_attrs) > 1:
                attrid = f"{self.registry_id}.{var_attrs['var_name']}"
            attrs["outputs"].append(_translate(attrid, var_attrs, TRANSLATABLE_ATTRS))
        return attrs

    def json(self, args=None):
        """Return a serializable dictionary representation of the class.

        Parameters
        ----------
        args : mapping, optional
            Arguments as passed to the call method of the indicator.
            If not given, the default arguments will be used when formatting the attributes.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.

        """
        names = ["identifier", "title", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out = self._format(out, args)

        # Format attributes
        out["outputs"] = [self._format(attrs, args) for attrs in self.cf_attrs]
        out["notes"] = self.notes

        # We need to deepcopy, otherwise empty defaults get overwritten!
        # All those tweaks are to ensure proper serialization of the returned dictionary.
        out["parameters"] = deepcopy(self.parameters)
        for param in out["parameters"].values():
            if param["default"] is _empty:
                param.pop("default")
            param["kind"] = param["kind"].value  # Get the int.
            if "choices" in param:  # A set is stored, convert to list
                param["choices"] = list(param["choices"])
        return out

    @classmethod
    def _format(
        cls,
        attrs: dict,
        args: dict = None,
        formatter: AttrFormatter = default_formatter,
    ):
        """Format attributes including {} tags with arguments.

        Parameters
        ----------
        attrs: dict
          Attributes containing tags to replace with arguments' values.
        args : dict, optional
          Function call arguments. If not given, the default arguments will be used when formatting the attributes.
        formatter : AttrFormatter
        """
        # Use defaults
        if args is None:
            args = {k: v["default"] for k, v in cls.parameters.items()}

        args.update(getattr(cls._indcompute, "_injected", {}))

        out = {}
        for key, val in attrs.items():
            mba = {"indexer": "annual"}
            # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
            for k, v in args.items():
                if isinstance(v, dict):
                    if v:
                        dk, dv = v.copy().popitem()
                        if dk == "month":
                            dv = "m{}".format(dv)
                        mba[k] = dv
                elif isinstance(v, units.Quantity):
                    mba[k] = "{:g~P}".format(v)
                elif isinstance(v, (int, float)):
                    mba[k] = "{:g}".format(v)
                else:
                    mba[k] = v

            if callable(val):
                val = val(**mba)

            out[key] = formatter.format(val, **mba)

            if key in cls._text_fields:
                out[key] = out[key].strip().capitalize()

        return out

    def _default_freq(self, **indexer):
        """Return default frequency."""
        if self.freq in ["D", "H"]:
            return indices.generic.default_freq(**indexer)
        return None

    def _mask(self, *args, **kwds):
        """Return whether mask for output values, based on the output of the `missing` method."""
        from functools import reduce

        indexer = kwds.get("indexer") or {}
        freq = kwds.get("freq") if "freq" in kwds else self._default_freq(**indexer)

        options = self.missing_options or OPTIONS[MISSING_OPTIONS].get(self.missing, {})

        # We flag periods according to the missing method.
        miss = (self._missing(da, freq, self.freq, options, indexer) for da in args)

        return reduce(np.logical_or, miss)

    # The following static methods are meant to be replaced to define custom indicators.
    @staticmethod
    def compute(*args, **kwds):
        """Compute the indicator.

        This would typically be a function from `xclim.indices`.
        """
        raise NotImplementedError

    @staticmethod
    def cfcheck(**das):
        """Compare metadata attributes to CF-Convention standards.

        When subclassing this method, use functions decorated using `xclim.core.options.cfcheck`.
        """
        return True

    @staticmethod
    def datacheck(**das):
        """Verify that input data is valid.

        When subclassing this method, use functions decorated using `xclim.core.options.datacheck`.

        For example, checks could include:
         - assert temporal frequency is daily
         - assert no precipitation is negative
         - assert no temperature has the same value 5 days in a row
        """
        return True


class Indicator2D(Indicator):
    """Indicator using two dimensions."""

    nvar = 2


class Daily(Indicator):
    """Indicator defined for inputs at daily frequency."""

    freq = "D"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_daily(da)


class Daily2D(Daily):
    """Indicator using two dimensions at daily frequency."""

    nvar = 2


class Hourly(Indicator):
    """Indicator defined for inputs at strict hourly frequency, meaning 3-hourly inputs would raise an error."""

    freq = "H"

    @staticmethod
    def datacheck(**das):  # noqa
        for key, da in das.items():
            datachecks.check_freq(da, "H")


def _parse_indice(indice: Callable, passed=None, **new_kwargs):
    """Parse an indice function and return corresponding elements needed for constructing an indicator.

    Parameters
    ----------
    indice : Callable
      A indice function, written according to xclim's guidelines.
    new_kwargs :
      Mapping from name to dicts containing the necessary info for injecting new keyword-only
      arguments into the indice_wrapper function. The meta dict can include (all optional):
      `default`, `description`, `annotation`.

    Returns
    -------
    indice_wrapper : callable
      A function with a new signature including the injected args in new_kwargs.
    docmeta : Mapping[str, str]
      A dictionary of the metadata attributes parsed in the docstring.
    params : Mapping[str, Mapping[str, Any]]
      A dictionary of metadata for each input parameter of the indice. The metadata dictionaries
      include the following entries: "default", "description", "kind" and, optionally, "choices" and "units".
      "kind" is one of the constants in :py:class:`xclim.core.utils.InputKind`.
    """
    # Base signature
    sig = signature(indice)
    passed = passed or {}

    # Update
    def _upd_param(param):
        # Required DataArray arguments receive their own name as new default
        #         + the Union[str, DataArray] annotation
        # Parameters with no default receive None
        if param.kind in [param.VAR_KEYWORD, param.VAR_POSITIONAL]:
            return param

        if param.annotation is DataArray:
            annot = Union[str, DataArray]
        else:
            annot = param.annotation

        default = passed.get(param.name, {}).get("default", param.default)
        if default is _empty:
            if param.annotation is DataArray:
                default = param.name
            else:
                default = None

        return Parameter(
            param.name,
            # We keep the kind, except we replace POSITIONAL_ONLY by POSITONAL_OR_KEYWORD
            max(param.kind, 1),
            default=default,
            annotation=annot,
        )

    # Parse all parameters, replacing annotations and default where needed and possible.
    new_params = list(map(_upd_param, sig.parameters.values()))

    # Injection
    for name, meta in new_kwargs.items():
        # ds argunent
        param = Parameter(
            name,
            Parameter.KEYWORD_ONLY,
            default=meta.get("default"),
            annotation=meta.get("annotation"),
        )

        if new_params[-1].kind == Parameter.VAR_KEYWORD:
            new_params.insert(-1, param)
        else:
            new_params.append(param)

    # Create new compute function to be wrapped in __call__
    indice_wrapper = copy_function(indice)
    indice_wrapper.__signature__ = new_sig = sig.replace(parameters=new_params)
    indice_wrapper.__doc__ = indice.__doc__

    # Docstring parsing
    parsed = parse_doc(indice.__doc__)

    # Extract params and pop those not in the signature.
    params = parsed.pop("parameters", {})
    for dropped in set(params.keys()) - set(new_sig.parameters.keys()):
        params.pop(dropped)

    if hasattr(indice, "in_units"):
        # Try to put units
        for var, ustr in indice.in_units.items():
            if var in params:
                params[var]["units"] = ustr

    # Fill default values and annotation in parameter doc
    for name, param in new_sig.parameters.items():
        if name in new_kwargs and "description" in new_kwargs[name]:
            params[name] = {"description": new_kwargs[name]["description"]}
        param_doc = params.setdefault(name, {"description": ""})
        param_doc["default"] = param.default
        param_doc["kind"] = infer_kind_from_parameter(param, "units" in param_doc)
        param_doc.update(passed.get(name, {}))

    return indice_wrapper, parsed, params


def build_indicator_module(
    name: str,
    objs: Mapping[str, Indicator],
    doc: Optional[str] = None,
) -> ModuleType:
    """Create a module from imported objects.

    The module is inserted as a submodule of `xclim.indicators`.

    Parameters
    ----------
    name : str
      New module name. If it already exists, the module is extended with the passed objects,
      overwriting those with same names.
    objs : dict
      Mapping of the indicators to put in the new module. Keyed by the name they will take in that module.
    doc : str
      Docstring of the new module. Defaults to a simple header. Invalid if the module already exists.

    Returns
    -------
    ModuleType
      A indicator module built from a mapping of Indicators.
    """
    from xclim import indicators

    if hasattr(indicators, name):
        if doc is not None:
            warnings.warn(
                "Passed docstring ignored when extending existing module.", stacklevel=1
            )
        out = getattr(indicators, name)
    else:
        doc = doc or f"{name.capitalize()} indicators\n" + "=" * (len(name) + 11)
        try:
            out = ModuleType(name, doc)
        except TypeError as err:
            raise TypeError(f"Module '{name}' is not properly formatted") from err
        indicators.__dict__[name] = out

    out.__dict__.update(objs)
    return out


def build_indicator_module_from_yaml(
    filename: PathLike,
    name: Optional[str] = None,
    base: Type[Indicator] = Daily,
    doc: Optional[str] = None,
    mode: str = "raise",
    realm: Optional[str] = None,
    keywords: Optional[str] = None,
    references: Optional[str] = None,
    notes: Optional[str] = None,
) -> ModuleType:
    """Build or extend an indicator module from a YAML file.

    The module is inserted as a submodule of `xclim.indicators`.

    Parameters
    ----------
    filename: PathLike
      Path to a YAML file.
    name: str, optional
      The name of the new or existing module, defaults to the name of the file.
      (e.g: `atmos.yml` -> `atmos`)
    base: Indicator subclass
      The Indicator subclass from which the new indicators are based. Superseeded by
      the class given in the yaml file or in individual indicator definitions (see submodule's doc).
    doc : str, optional
      The docstring of the new submodule. Defaults to a very minimal header with the submodule's name.
    mode: {'raise', 'warn', 'ignore'}
      How to deal with broken indice definitions.
    realm: str, optional
    keywords: str, optional
       Comma separated keywords.
    references: str, optional
        Source citations.
    notes: str, optional
      Other indicator attributes that would apply to all indicators in this module.
      Values given here are overridden by the ones given in individual definition, but
      they override the ones given at top-level in the YAMl file.

    Returns
    -------
    ModuleType
      A submodule of `xclim.indicators`.

    See also
    --------
    The doc of :py:mod:`xclim.core.indicator` and of :py:func:`build_module`.
    """
    # Read YAML file
    filepath = Path(filename)
    with filepath.open() as f:
        yml = safe_load(f)

    # Load values from top-level in yml.
    # Priority of arguments differ.
    module_name = name or yml.get("module", filepath.stem)
    default_base = registry.get(yml.get("base"), base)
    doc = doc or yml.get("doc")

    # Module-wide default values for some attributes
    defkwargs = {
        # We can override the module of indicators in their init (weird but cool)
        # This way, submodule indicators are prefixed with the module name in the registry.
        "module": module_name,
        # Other default argument, only given in case the indicator definition does not give them.
        "realm": realm or yml.get("realm"),
        "keywords": keywords or yml.get("keywords"),
        "references": references or yml.get("references"),
        "notes": notes or yml.get("notes"),
    }
    # Parse the indicators:
    mapping = {}
    for identifier, data in yml["indices"].items():
        # clix-meta has illegal characters in the identifiers.
        clean_id = identifier.replace("{", "").replace("}", "")
        try:
            if "base" in data:
                base = registry[data["base"].upper()]
            else:
                base = default_base
            mapping[clean_id] = base.from_dict(data, clean_id, **defkwargs)
        except Exception as err:
            msg = f"Constructing {identifier} failed with {err!r}"
            if mode == "ignore":
                logging.info(msg)
            elif mode == "warn":
                warnings.warn(msg)
            else:  # mode == "raise"
                raise ValueError(msg) from err

    # Construct module
    return build_indicator_module(module_name, objs=mapping, doc=doc)
