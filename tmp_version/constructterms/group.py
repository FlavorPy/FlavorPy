class Group:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class AbelianGroup(Group):
    """
    An abelian group Z_order

    :param name: The name of this abelian group
    :type name: bool, optional
    :param order: The order N of this abelian group Z_N
    :type order: int, optional, default:1
    """
    def __init__(self, name, order=1):
        self.order = order
        super().__init__(name)

    def make_product(self, chargeA: int, chargeB: int) -> int:
        return (chargeA + chargeB) % self.order

    def is_desired(self, charge_field, charge_desired) -> bool:
        if charge_field == charge_desired:
            return True
        else:
            return False


class NonAbelianGroup(Group):
    """
    A non-abelian group

    :param name: The name of this non-abelian group
    :type name: str
    :param gapid: The Gap-Id of this group. For now this has no relevance. The idea is to later add a function that
        can obtain the 'tensor_products' out of gap or sage.
    :type gapid: list, optional
    :param representations: A list of all (relevant) representations of this non-abelian group
    :type representations: list, optional
    :param tensor_products: A matrix that contains the tensor products.
    :type tensor_products: list, optional
    :param clebsches: For now this has no relevance. The idea would be to later add a function to the package that
        can give you the Clebsch-Gordans so that a multiplication of actual matrix representations is possible.
    :type clebsches: dict, optional
    """
    def __init__(self, name, gapid=None, representations=None, tensor_products=None, clebsches=None):
        if gapid is None:
            gapid = []
        if representations is None:
            representations = []
        if tensor_products is None:
            tensor_products = [[[]]]
        if clebsches is None:
            clebsches = {}

        self.gapid = gapid
        self.representations = representations
        self.tensor_products = tensor_products
        self.clebsches = clebsches
        super().__init__(name)

    def make_product(self, repA: list, repB: list) -> list:
        """
        Calculates the resulting representations of the tensor product between repA and repB

        :param repA: A list of irreducible representations. E.g. ['3_1', '3_2'].
            Entries have to be in 'self.representations'!
        :type repA: list
        :param repB: A list of irreducible representations. E.g. ['3_1', '3_2']. Entries have to be in
            'self.representations'!
        :type repB: list
        :return: A list of irreducible representations.
        :rtype: list
        """
        repC = []
        for irrepA in repA:
            for irrepB in repB:
                repC.extend(self.tensor_products[self.representations.index(irrepA)]
                                                [self.representations.index(irrepB)])
        return repC

    def make_product_components(self, compA: dict, compB: dict) -> dict:
        """
        Calculates the components of the tensor product of compA and compB, while using the Clebsch-Gordans
        that have been defined for the group.

        :param compA: The components of representation A. E.g. {'2':[['A1','A2']]}
        :type compA: dict
        :param compB: The components of representation B.
            E.g. {'3':[['B1','B2','B3],['B4','B5','B6']]} for a 3 + 3 representation
        :type compB: dict
        :return: The components of the product. E.g. {'1':[['A1 B2 + A2 B1']], '2':[['A1 B1 - A2 B2','A1 B3 + A2 B2']]}
        :rtype: dict
        """
        compC = {}
        for irrepA in compA:
            for irrepB in compB:
                if str(irrepA)+' x '+str(irrepB) in self.clebsches:
                    irrepAxB = str(irrepA)+' x '+str(irrepB)
                    local_compA, local_compB = compA, compB
                elif str(irrepB)+' x '+str(irrepA) in self.clebsches: # interchange A and B if only 'B x A' in clebsches
                    irrepAxB = str(irrepB)+' x '+str(irrepA)
                    irrepA, irrepB = irrepB, irrepA
                    local_compA, local_compB = compB, compA
                else:
                    raise KeyError(f'''The tensor product '{str(irrepA)} x {str(irrepB)}' is not defined 
                                       in the clebsches of the group {str(self.name)}!''')
                for entryA in local_compA[irrepA]:
                    # Make brackets around name of component if it contains a '+' or '-'
                    if any(x.__contains__('+') for x in entryA) or any(x.__contains__('+') for x in entryA):
                        entryA = ['('+x+')' for x in entryA]
                    for entryB in local_compB[irrepB]:
                        # Make brackets around name of component if it contains a '+' or '-'
                        if any(x.__contains__('+') for x in entryB) or any(x.__contains__('+') for x in entryB):
                            entryB = ['(' + x + ')' for x in entryB]
                        for irrepC in self.clebsches[irrepAxB]:
                            for clebsch_mat in self.clebsches[irrepAxB][irrepC]:
                                entryC = []
                                for clebsch_line in clebsch_mat:
                                    entryC_component = ''
                                    for i in range(len(clebsch_line)):
                                        coef = clebsch_line[i]
                                        if str(coef) != "0":
                                            if str(coef).strip() == "" or str(coef).strip()[0] != "-":
                                                entryC_component += '+'
                                            if str(coef) == "1" or str(coef) == "-1":
                                                coef = str(coef)[:-1]
                                            else:
                                                entryC_component += ' '
                                            entryC_component += str(coef)+' '
                                            entryC_component += str(entryA[int(i/len(entryB))])+' '
                                            entryC_component += str(entryB[int(i % len(entryB))])+' '
                                    entryC_component = entryC_component.strip(" +")
                                    entryC.append(entryC_component)
                                if irrepC in compC:
                                    compC[irrepC] += [entryC]
                                else:
                                    compC[irrepC] = [entryC]
        return compC

    def is_desired(self, rep_field: list, rep_desired: list) -> bool:
        """
        Note that this function does not check if repA and repB are the same, but rather if any of the reps in
        'rep_desired' is contained in 'rep_field' !!
        """
        for irrep_desired in rep_desired:
            if irrep_desired in rep_field:
                return True
        return False


class U1Group(Group):
    """
    A U(1) group

    :param name: The name of the group
    :type name: str
    """
    def __init__(self, name):
        super().__init__(name)

    def make_product(self, chargeA, chargeB):
        return chargeA + chargeB

    def is_desired(self, charge_field, charge_desired) -> bool:
        if charge_field == charge_desired:
            return True
        else:
            return False
