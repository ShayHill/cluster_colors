"""
This type stub file was generated by pyright.
"""

from distutils.command.build_ext import build_ext as old_build_ext

""" Modified version of build_ext that handles fortran source files.

"""
class build_ext(old_build_ext):
    description = ...
    user_options = ...
    help_options = ...
    boolean_options = ...
    def initialize_options(self): # -> None:
        ...
    
    def finalize_options(self): # -> None:
        ...
    
    def run(self): # -> None:
        ...
    
    def swig_sources(self, sources, extensions=...):
        ...
    
    def build_extension(self, ext):
        ...
    
    def get_source_files(self): # -> list[Unknown]:
        ...
    
    def get_outputs(self): # -> list[Unknown]:
        ...
    


