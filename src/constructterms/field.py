class Field:
    """
    A field or representation that has a name und is charged under some symmetry-groups.
    """
    def __init__(self, name, charges={}):
        """
        Field
        ----------
        :param name: str
            The name of the field / representation
        :param charges: dict, optional
            The charges / irreps under the Groups. Has the form {Group1: charge1, Group2: charge2}, where 'Group1' and
            'Group2' have to be an Object of the 'Group' class. Note that abelian groups have integer charges,
            U(1) groups have integer or float charges and non-abelian groups have a list of one or more irreps, e.g.
            {Non_Abelian_Group: ['3_1','3_2']}.
        """
        self.name = name
        self.charges = charges

    def times(self, other_field):
        """
        Calculates the tensor product of 'self' and 'other_field'.
        ----------
        :param other_field: The field that you want to multiply 'self' with. Has to be of 'Field'-class!
        :return: An object of the 'Field'-class that represents the tenosr product of 'self' and 'other_field'.
        """
        if self.charges.keys() != other_field.charges.keys():
            raise KeyError('''The Field that you are multiplying with is not charged under the same symmetries! Make sure
                           that both fields have the same symmetries in the 'charges'-dictionary!''')
        return Field(self.name+' '+other_field.name,
                     charges={group: group.make_product(self.charges[group], other_field.charges[group])
                              for group in self.charges})

    def is_desired(self, desired_field, print_cause=False) -> bool:
        """
        Check if 'self' is charged in the same way as 'desired_field' under all symmetries. For non-abelian symmetries
        it checks, if 'self' contains at least one of the irreps of 'desired_field'. Use this for example to check if
        a lagrangian-term is invariant.
        :param desired_field: Compare the charges of 'self' to this field. Has to be of 'Field'-class!
        :param print_cause: bool, optional
            If 'True' it prints which symmetry is causing the end-result to be 'False'
        :return: bool
        """
        if self.charges.keys() != desired_field.charges.keys():
            raise KeyError('''The Field that you are comparing with is not charged under the same symmetries! Make sure
                           that both fields have the same symmetries in the 'charges'-dictionary!''')
        result = all([group.is_desired(self.charges[group], desired_field.charges[group]) for group in self.charges])
        if print_cause is True and result is False:
            for group in self.charges:
                if not group.is_desired(self.charges[group], desired_field.charges[group]):
                    print('The charge/irreps of your field under the group '+group.name+' is not the desired one')
        return result



