## For an illustrative example of how to use this code, evaluate the following: #########################################
#
#import ConstructTerms as CT
#
## abelian symmetries should be defined as follows: 'abelian':{'Z3':[1,3]} for a charge of 1 of a Z_3 symmetry...
#L = {'name':'L', 'non-abelian':{'A4':['3']}, 'abelian':{}, 'U1':{'U1R':1, 'modweight':0, 'U1Y':-1/2}}
#E1 = {'name':'E1', 'non-abelian':{'A4':['1']}, 'abelian':{}, 'U1':{'U1R':1, 'modweight':0, 'U1Y':1}}
#E2 = {'name':'E2', 'non-abelian':{'A4':['1pp']}, 'abelian':{}, 'U1':{'U1R':1, 'modweight':0, 'U1Y':1}}
#E3 = {'name':'E3', 'non-abelian':{'A4':['1p']}, 'abelian':{}, 'U1':{'U1R':1, 'modweight':0, 'U1Y':1}}
#N = {'name':'N', 'non-abelian':{'A4':['3']}, 'abelian':{}, 'U1':{'U1R':1, 'modweight':0, 'U1Y':0}}
#Hd = {'name':'Hd', 'non-abelian':{'A4':['1']}, 'abelian':{}, 'U1':{'U1R':0, 'modweight':0, 'U1Y':-1/2}}
#Hu = {'name':'Hu', 'non-abelian':{'A4':['1']}, 'abelian':{}, 'U1':{'U1R':0, 'modweight':0, 'U1Y':1/2}}
#Phi = {'name':'Phi', 'non-abelian':{'A4':['3']}, 'abelian':{}, 'U1':{'U1R':0, 'modweight':0, 'U1Y':0}}
#zeta = {'name':'zeta', 'non-abelian':{'A4':['3']}, 'abelian':{}, 'U1':{'U1R':2, 'modweight':-2, 'U1Y':0}}
#Y = {'name':'Y', 'non-abelian':{'A4':['3']}, 'abelian':{}, 'U1':{'U1R':0, 'modweight':2, 'U1Y':0}}
#trivial = {'name':'', 'non-abelian':{'A4':['1']}, 'abelian':{}, 'U1':{'U1R':0, 'modweight':0, 'U1Y':0}}
#total = {'name':'total', 'non-abelian':{'A4':['1']}, 'abelian':{}, 'U1':{'U1R':2, 'modweight':0, 'U1Y':0}}
#Fields = [L, E1, E2, E3, N, Hd, Hu, Phi, zeta, Y, trivial]
#
#A4Reps = ['1', '1p', '1pp', '3']
#A4TensorProducts = [[['1'], ['1p'], ['1pp'], ['3']],
#                    [['1p'], ['1pp'], ['1'], ['3']],
#                    [['1pp'], ['1'], ['1p'], ['3']],
#                    [['3'], ['3'], ['3'], ['1','1p','1pp','3','3']]]
#
#kwargs = {'Order':4, 'Reps':A4Reps, 'TensorProducts':A4TensorProducts}
#terms = CT.ConstructAllProducts(Fields, trivial, **kwargs)
#terms = CT.FilterInvariant(terms, total,**kwargs)
#print('The '+str(len(terms))+' terms up to order '+str(kwargs['Order'])+' that are allowed by the symmetries are:')
#for term in terms:
#    print(term['name'])
#
#print('One can also check whether specific term is allowed or not, e.g.')
#CT.CheckIfAllowed(CT.Product(CT.Product(E1,Hd, **kwargs), L, **kwargs), total, printcause=True, **kwargs)
#
## Code by alex ;) ######################################################################################################
# Possible ways to extend the code:
# -> add SU(2) representations. For now, U(1)_Y was sufficient
#########################################################################################################################

from itertools import combinations_with_replacement

def Product(A, B, Reps=[], TensorProducts=[[]], **kwargs):
    # non-abelian
    key = list(A['non-abelian'])[0]
    if len(A['non-abelian'].keys()) > 1:
        print('Only calculations with 1 non-abelian symmetry are possible!')
        # make a warning or cause calculation to stop
    res = []
    for repA in A['non-abelian'][key]:
        for repB in B['non-abelian'][key]:
            res.extend(TensorProducts[Reps.index(repA)][Reps.index(repB)])
    nonab = {key:res}
    
    # abelian
    ab = {}
    if A['abelian'].keys() != B['abelian'].keys():
        print('Problem with abelian symmetry: Your fields do not have the same symmetries!')
    for key in A['abelian']:
        if A['abelian'][key][1] != B['abelian'][key][1]:
            print('Problem with abelian symmetry called '+key+'! Not all fields have the same n of this Z_n symmetry!')
            abA, abB, n = A['abelian'][key][0], B['abelian'][key][0], A['abelian'][key][1]
        ab[key] = [np.mod(abA+abB, n), n]
        
    # U1
    u1 = {}
    if A['U1'].keys() != B['U1'].keys():
        print('Problem with U1 symmetry: Your fields do not have the same symmetries!')
    for key in A['U1']:
        u1A, u1B = A['U1'][key], B['U1'][key]
        u1[key] = u1A+u1B
    
    
    return {'name':A['name']+' '+B['name'], 'non-abelian':nonab, 'abelian':ab, 'U1':u1}
    
def CheckIfAllowed(term, total, docheck=True, printcause=False, **kwargs):
    key = list(total['non-abelian'])[0]
    # some checks:
    if docheck==True:
        if term['abelian'].keys() != total['abelian'].keys():
            print('Problem with abelian symmetry: Your term does not have the same symmetries as '+total['name']+'!')
        if term['U1'].keys() != total['U1'].keys():
            print('Problem with U1 symmetry: Your term does not have the same symmetries as \''+total['name']+'\'!')
        if len(term['non-abelian'].keys()) > 1:
            print('Only calculations with 1 non-abelian symmetry are possible!')
        if len(total['non-abelian'][key])<1:
            print('Filter Invariants: There are several representations allowed for non-abelian symmetry in '+total['name']+'?!')
    # Checking
    if total['non-abelian'][key][0] in term['non-abelian'][key]:
        nonabcheck = [True]
    else:
        nonabcheck = [False]
    abcheck = []
    for key in total['abelian']:
        if total['abelian'][key] == term['abelian'][key]:
            abcheck.append(True)
        else:
            abcheck.append(False)
    u1check = []
    for key in total['U1']:
        if total['U1'][key] == term['U1'][key]:
            u1check.append(True)
        else:
            u1check.append(False)   
    final = all(el for el in nonabcheck+abcheck+u1check)
    
    if printcause==True:
        if final==False:
            print('Coupling '+term['name']+' not allowed because... non-abelian: '+str(nonabcheck)+', abelian: '+str(abcheck)+', U1: '+str(u1check))
    return final
    
def ConstructAllProducts(Fields, trivial, Order=4, **kwargs):
    combs = list(combinations_with_replacement(Fields, Order))
    Terms = []
    for comb in combs:
        term = trivial
        for field in comb:
            term = Product(term, field, **kwargs)
        Terms.append(term)
    return Terms
    
def FilterInvariant(Terms, total, **kwargs):
    Invariants = []
    for term in Terms:
        if CheckIfAllowed(term, total, docheck=False, **kwargs):
            Invariants.append(term)
    return Invariants
