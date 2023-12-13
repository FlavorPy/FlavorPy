"""
ConstructTerms

ConstructTerms allows you to make tensor products of fields charged under several symmetries.
It can be used for example to find all invariant terms of a Lagrangian/Superpotential
"""
from .field import Field
from .group import Group, NonAbelianGroup, AbelianGroup, U1Group
from .groups import groups
from .calculations import list_allowed_terms
