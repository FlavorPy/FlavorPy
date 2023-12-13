class Field:
    """
    A field or representation that has a name und is charged under some symmetry-groups.

    :param name: The name of the field / representation
    :type name: str
    :param charges: The charges / irreps under the Groups. Has the form {Group1: charge1, Group2: charge2}, where
        'Group1' and 'Group2' have to be an Object of the :py:meth:`~constructterms.group.Group` class.
        Note that abelian groups have integer charges, U(1) groups have integer or float charges and non-abelian groups
        have a list of one or more irreps, e.g.::
            charges = {Abelian_Group: 2, U1_Group: 0.5, Non_Abelian_Group: ['3_1','3_2']}.
    :type charges: dict, optional
    :param components: The single components of a field. E.g. if the field is a '3' representation under A4,
        it would be::
            'components = {A4: {'3': [['x1', 'x2', 'x3']]}}'.
    :type components: dict, optional
    """
    def __init__(self, name, charges=None, components=None):
        if charges is None:
            charges = {}
        if components is None:
            components = {}

        self.name = name
        self.charges = charges
        self.components = components

    def __repr__(self):
        return self.name

    def times(self, other_field):
        """
        Calculates the tensor product of 'self' and 'other_field'.

        :param other_field: The other field that you want to multiply this field with.
        :type other_field: :py:meth:`~constructterms.field.Field`
        :return: A field that represents the tensor product of 'self' and 'other_field'.
        :rtype: :py:meth:`~constructterms.field.Field`
        """
        # Do tensor products
        if self.charges.keys() != other_field.charges.keys():
            raise KeyError('''The Field that you are multiplying with is not charged under the same symmetries! 
                           Make sure that both fields have the same symmetries in the 'charges'-dictionary!''')
        new_charges = {group: group.make_product(self.charges[group], other_field.charges[group])
                       for group in self.charges}
        # Calculate components with Clebsch-Gordans
        if self.components.keys() != other_field.components.keys():
            raise KeyError('''The Field that you are multiplying with does not have components under the same 
                           symmetries! Make sure that both fields have the same symmetries in the 'charges'-dictionary!
                           ''')
        new_components = {group: group.make_product_components(self.components[group], other_field.components[group])
                          for group in self.components}
        # Return result
        return Field(self.name + ' ' + other_field.name, charges=new_charges, components=new_components)

    def is_desired(self, desired_field, print_cause=False, ignore=None) -> bool:
        """
        Check if 'self' is charged in the same way as 'desired_field' under all symmetries. For non-abelian symmetries
        it checks, if 'self' contains at least one of the irreps of 'desired_field'. Use this for example to check if
        a lagrangian-term is invariant.

        :param desired_field: Compare the charges of 'self' to this field. Has to be of 'Field'-class!
        :type desired_field: :py:meth:`~constructterms.field.Field`
        :param ignore: List here any symmetry that you do not want to compare to the desired field.
        :type ignore: list, optional
        :param print_cause: If 'True' it prints which symmetry is causing the end-result to be 'False'
        :type print_cause: bool, optional
        :return: True or False
        :rtype: bool
        """
        if ignore is None:
            ignore = []
        if self.charges.keys() != desired_field.charges.keys():
            raise KeyError('''The Field that you are comparing with is not charged under the same symmetries! Make sure
                           that both fields have the same symmetries in the 'charges'-dictionary!''')
        result = all([group.is_desired(self.charges[group], desired_field.charges[group])
                      for group in self.charges if group not in ignore])
        if print_cause is True and result is False:
            for group in self.charges:
                if group not in ignore:
                    if not group.is_desired(self.charges[group], desired_field.charges[group]):
                        print('The charge/irreps of your field under the group '+group.name+' is not the desired one')
        return result
