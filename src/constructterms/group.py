class Group:
    def __init__(self, name):
        self.name = name


class AbelianGroup(Group):
    def __init__(self, name, order=1):
        """
        An abelian group Z_order
        ----------
        :param name: str
            The name of this abelian group
        :param order: int, optional, default:1
            The order N of this abelian group Z_N
        """
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
    def __init__(self, name, gapid=[], representations=[], tensor_products=[[[]]], clebsches=[]):
        """
        A non-abelian group
        ----------
        :param name: str
            The name of this non-abelian group
        :param gapid: list, optional
            The Gap-Id of this group. For now this has no relevance. The idea is to later add a function that can obtain
            the 'tensor_products' out of gap or sage.
        :param representations: list, optional
            A list of all (relevant) representations of this non-abelian group
        :param tensor_products: list, optional
            A matrix that contains
        :param clebsches:  list, optional
            For now this has no relevance. The idea would be to later add a function to the package that can give you
            the Clebsch-Gordans so that a multiplication of actual matrix representations is possible.
        """
        self.gapid = gapid
        self.representations = representations
        self.tensorproducts = tensor_products
        self.clebsches = clebsches
        super().__init__(name)

    def make_product(self, repA: list, repB: list) -> list:
        repC = []
        for irrepA in repA:
            for irrepB in repB:
                repC.extend(self.tensorproducts[self.representations.index(irrepA)]
                                               [self.representations.index(irrepB)])
        return repC

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
    def __init__(self, name):
        """
        A U(1) group
        ----------
        :param name: str
            The name of the group
        """
        super().__init__(name)

    def make_product(self, chargeA, chargeB):
        return chargeA + chargeB

    def is_desired(self, charge_field, charge_desired) -> bool:
        if charge_field == charge_desired:
            return True
        else:
            return False
