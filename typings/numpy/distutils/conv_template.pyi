"""
This type stub file was generated by pyright.
"""

"""
takes templated file .xxx.src and produces .xxx file  where .xxx is
.i or .c or .h, using the following template rules

/**begin repeat  -- on a line by itself marks the start of a repeated code
                    segment
/**end repeat**/ -- on a line by itself marks it's end

After the /**begin repeat and before the */, all the named templates are placed
these should all have the same number of replacements

Repeat blocks can be nested, with each nested block labeled with its depth,
i.e.
/**begin repeat1
 *....
 */
/**end repeat1**/

When using nested loops, you can optionally exclude particular
combinations of the variables using (inside the comment portion of the inner loop):

 :exclude: var1=value1, var2=value2, ...

This will exclude the pattern where var1 is value1 and var2 is value2 when
the result is being generated.


In the main body each replace will use one entry from the list of named replacements

 Note that all #..# forms in a block must have the same number of
   comma-separated entries.

Example:

    An input file containing

        /**begin repeat
         * #a = 1,2,3#
         * #b = 1,2,3#
         */

        /**begin repeat1
         * #c = ted, jim#
         */
        @a@, @b@, @c@
        /**end repeat1**/

        /**end repeat**/

    produces

        line 1 "template.c.src"

        /*
         *********************************************************************
         **       This file was autogenerated from a template  DO NOT EDIT!!**
         **       Changes should be made to the original source (.src) file **
         *********************************************************************
         */

        #line 9
        1, 1, ted

        #line 9
        1, 1, jim

        #line 9
        2, 2, ted

        #line 9
        2, 2, jim

        #line 9
        3, 3, ted

        #line 9
        3, 3, jim

"""
__all__ = ['process_str', 'process_file']
global_names = ...
header = ...
def parse_structure(astr, level): # -> list[Unknown]:
    """
    The returned line number is from the beginning of the string, starting
    at zero. Returns an empty list if no loops found.

    """
    ...

def paren_repl(obj): # -> LiteralString:
    ...

parenrep = ...
plainrep = ...
def parse_values(astr): # -> list[str]:
    ...

stripast = ...
named_re = ...
exclude_vars_re = ...
exclude_re = ...
def parse_loop_header(loophead): # -> list[Unknown]:
    """Find all named replacements in the header

    Returns a list of dictionaries, one for each loop iteration,
    where each key is a name to be substituted and the corresponding
    value is the replacement string.

    Also return a list of exclusions.  The exclusions are dictionaries
     of key value pairs. There can be more than one exclusion.
     [{'var1':'value1', 'var2', 'value2'[,...]}, ...]

    """
    ...

replace_re = ...
def parse_string(astr, env, level, line): # -> LiteralString:
    ...

def process_str(astr): # -> str:
    ...

include_src_re = ...
def resolve_includes(source): # -> list[Unknown]:
    ...

def process_file(source): # -> str:
    ...

def unique_key(adict): # -> LiteralString:
    ...

def main(): # -> None:
    ...

if __name__ == "__main__":
    ...
