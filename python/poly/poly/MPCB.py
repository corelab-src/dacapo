import numpy as np
import torch
import torch.nn.functional as F
import einops
import hecate as hc

Empty = hc.Empty

def cint (i) : 
    return int(np.ceil(i))

def fint (i) : 
    return int(np.floor(i))

def maximum(a,b) : 
    return torch.maximum (a,b)

# def roll (A, i) :
#     return torch.roll (A, i)

def roll(A, i) :
    return A.rotate(-i)

def GenPoly (treeStr, coeffStr, length, scale = 1.0) : 

    tree = [token.strip().split(" ") for token in treeStr]
    tree = [[int(istr) for istr in istrl] for istrl in tree]

    coeff = [float(token.strip())/scale for token in coeffStr]
    cheby_mish = np.polynomial.Chebyshev(np.array(coeff, dtype=np.double))


    newtree = [[num for num in arr ] for arr in tree]
    newtree[0][0] = cheby_mish

    for i, cousins in enumerate(tree) : 
        new_index = []
        for j, divisor in enumerate(cousins) :
            if (divisor <= 0) :
                pass
            else :
                divisor = np.polynomial.Chebyshev([0] * divisor + [1])
                newtree[i+1][2*j+1] = newtree[i][j] // divisor
                newtree[i+1][2*j] = newtree[i][j] %divisor

    calc_order = [ (i, j, divisor, newtree[i][j]) for i, cousins in enumerate(tree) for j, divisor in enumerate(cousins) if divisor >= 0]
    calc_order = sorted(calc_order, reverse=True, key=lambda x: x[0])
#     print (calc_order)

    def polynomial (x) :
        babyTs = []
        giantTs = { 0 : 1, 1 : x}
        tmpTs = {}
        for i in range(1, fint(np.log2(length))) :
            idx = pow(2, i)
            pre_idx = pow(2,i-1)
            giantTs[idx] = 2 * giantTs[pre_idx]  * giantTs[pre_idx] + -1
        babyTs = [x]
        for i in range(1, fint(np.log2(length))) :
            idx = pow(2, i)
            babyAdd = [2 * poly * giantTs[idx] for poly in babyTs]
            #bug
            sdfs = [ new+old for new, old in zip (babyAdd, reversed(babyTs))]
            babyAdd = [ new-old for new, old in zip (babyAdd, reversed(babyTs))]
            babyTs = babyTs + babyAdd
        tmpPoly = {}
        for i,j, deg, leaf in calc_order :
            if (deg == 0) :
                poly = 0
                for k in range (length//2 ) : 
                    if (len(leaf.coef) > 2*k+1 ) : 
                        poly += leaf.coef[2*k+1] * babyTs[k]
                tmpPoly[(i,j)] = poly
            else :
                if not deg in giantTs : 
                    giantTs[deg] = 2* giantTs[deg//2] * giantTs[deg//2] + -1
                tmpPoly[(i,j)] = tmpPoly[(i+1, 2*j+1)] * giantTs[deg] + tmpPoly[(i+1, 2*j)]

        return tmpPoly[(0,0)]
    return polynomial

def shapeClosure(nt, bb, fh, fw, s , hi, ho, wi, wo, ni, no, ci, co, ki, ko, ti, to, pi, po, q) :
    # print (fh, fw, s , hi, ho, wi, wo, ni, no, ci, co, ki, ko, ti, to, pi, po, q)

    
    def MultParPack ( A) :
        A = A/bb
        A = F.pad (A[0], (0, 0, 0, 0, 0, ki*ki*ti - ci ), mode = 'constant', value = 0)
        A = einops.rearrange(A, '(ti s1 s2) h w -> (ti h s1 w s2)',  s1 = ki, s2 = ki)
        A = F.pad (A, (0, ni * nt//pi - A.shape[0]), mode = 'constant', value=0)
        A = einops.repeat (A, '(ni a) -> ni (pi a)' , ni = ni, pi = pi)
        C = np.full( (ni) ,Empty(), dtype = object)
        for i in range(ni) : 
             C[i] = A[i, :]
        # Temporary -
        # A = einops.rearrange (A, '(ni a) -> ni a' , ni = ni)
        return A

    def OutPack ( A) :
        A = A/bb
        A = F.pad (A[0], (0, 0, 0, 0, 0, ko*ko*to - co ), mode = 'constant', value = 0)
        A = einops.rearrange(A, '(to s1 s2) h w -> (to h s1 w s2)',  s1 = ko, s2 = ko)
        A = F.pad (A, (0, no * nt//po - A.shape[0]), mode = 'constant', value=0)
        A = einops.repeat (A, 'a -> (po a)' , po = po)
        A = einops.rearrange (A, '(no a) -> no a' , no = no)
        return A

    def ParMultWgt (U) : 
        U = F.pad (U, (0, 0, 0, 0, 0, ki*ki*ti - ci, 0, q*pi - co ), mode = 'constant', value = 0)
        U = einops.repeat(U, '(q pi) (ti s1 s2) fh fw -> (fh fw q pi) (ti h s1 w s2)', q = q, pi=pi, s1 = ki, s2 = ki, h = hi, w= wi)
        U = F.pad (U, (0, ni * nt//pi- U.shape[1] ), mode = 'constant', value=0)
        U = einops.rearrange(U, '(fh fw q pi) (ni A) -> ni q fh fw (pi A)', q = q, pi = pi, fh = fh, fw= fw, ni=ni)
        M = torch.ones(hi, wi, dtype=torch.double)
        M = F.pad (M, (((fw-1)//2), ((fw-1)//2), ((fh-1)//2), ((fh-1)//2)), mode = 'constant', value=0)
        M = torch.stack([ M[i:i+hi,j:j+wi] for i in range (fh) for j in range(fw)])
        M = einops.repeat(M, '(fh fw) h w -> ni q fh fw (kk h s1 w s2)', fh = fh, fw = fw, q=q, kk = nt//(hi * ki * wi * ki), s1 = ki, s2 = ki, ni=ni)
        # U = U * M.cuda()
        U = U * M
        return U

    def DwMultWgt (U) : 
        U = F.pad (U, (0, 0, 0, 0, 0, 0, 0, ki*ki*ti - ci), mode = 'constant', value = 0)
        U = einops.repeat(U, '(ti s1 s2) i fh fw -> fh (fw i) (ti h s1 w s2)', s1 = ki, s2 = ki, h = hi, w= wi)
        U = F.pad (U, (0, ni * nt//pi- U.shape[2] ), mode = 'constant', value=0)
        U = einops.repeat(U, 'fh fw (ni A) -> ni fh fw (pi A)',pi = pi, fh = fh, fw= fw, ni=ni)
        M = torch.ones(hi, wi, dtype=torch.double)
        M = F.pad (M, (((fw-1)//2), ((fw-1)//2), ((fh-1)//2), ((fh-1)//2)), mode = 'constant', value=0)
        M = torch.stack([ M[i:i+hi,j:j+wi] for i in range (fh) for j in range(fw)])
        M = einops.repeat(M, '(fh fw) h w -> ni fh fw (kk h s1 w s2)', fh = fh, fw = fw, kk = nt//(hi * ki * wi * ki), s1 = ki, s2 = ki, ni=ni)
        # U = U * M.cuda()
        U = U * M
        return U


    def SumSlots (A, m, p) :
        #B = torch.zeros( (fint (np.log2(m)) +1, nt), dtype = torch.double)
        B = np.full( (fint (np.log2(m)) +1, ) ,Empty(), dtype = object)

        B[0] = A
        for j in range(1, fint(np.log2(m))+1) :
            B[j] = B[j-1] + roll (B[j-1], -pow(2, j-1)* p)
        C = B[fint(np.log2(m))]
        for j in range(0, fint (np.log2(m))) :
            if ((m//pow(2, j))%2) == 1 :
                C = C + roll (B[j], -(m//pow(2, j+1)) * pow(2, j+1)* p)
        return C

    def Selecting () : 
        S = torch.eye (co, ko* ko * to, dtype=torch.double )
        S = einops.repeat(S, 'co (to s1 s2 ) -> co (to ho s1 wo s2)', co = co, s1 = ko, s2 = ko, to = to, ho = ho, wo=wo)
        S = F.pad (S, (0, no* nt- S.shape[1]), mode = 'constant', value =0)
        S = einops.rearrange(S, 'co (no A) -> no co A', co = co, no = no)
        return S

    def ParBNConst (C) :
        C = F.pad (C, (0, ko*ko*to - co), mode = 'constant', value = 0)
        C = einops.repeat(C, '(to s1 s2) -> (to ho s1 wo s2)', to = to, s1 = ko, s2 = ko, ho = ho, wo = wo)
        C = F.pad ( C, (0, no* nt//po - C.shape[0]), mode = 'constant', value = 0)
        C = einops.repeat(C, '(no c) -> no (po c)', po = po, no=no)
        return C

    def ParInBNConst (C) :
        C = F.pad (C, (0, ki*ki*ti - ci), mode = 'constant', value = 0)
        C = einops.repeat(C, '(ti s1 s2) -> (ti hi s1 wi s2)', ti = ti, s1 = ki, s2 = ki, hi = hi, wi = wi)
        C = F.pad ( C, (0, ni* nt//pi - C.shape[0]), mode = 'constant', value = 0)
        C = einops.repeat(C, '(ni c) -> ni (pi c)', pi = pi, ni=ni)
        return C

    def DownSelecting() :
        S = torch.eye ( ki * ti, ki * ti, dtype=torch.double)
        S = einops.repeat (S, ' (ki1 ti1) (ki2 ti2) -> ki1 ti1 ti2 hi s1 ki2 wi s2 ki', ki1 = ki, ti1 = ti, ki2 = ki, ti2 = ti, hi = hi//s, wi = wi//s, ki = ki, s1 = 1, s2 =1)
        S = F.pad (S, (0, 0, 0, s-1, 0, 0, 0, 0, 0, s-1), mode = 'constant', value = 0)
        S = einops. rearrange (S, 'ki1 ti1 ti2 hi s1 ki2 wi s2 ki -> ki1 ti1 ti2 (hi s1) ki2 (wi s2) ki')
        S = F.pad (S, (0, 0, 0, wi%s, 0, 0, 0, hi%s), mode = 'constant', value = 0)
        S = einops. rearrange (S, 'ki1 ti1 ti2 hi ki2 wi ki -> ki1 ti1 (ti2 hi ki2 wi ki)')
        S = F.pad (S, (0, ni *nt- S.shape[2]), mode = 'constant', value = 0)
        S = einops. rearrange (S, 'ki ti (ni A) -> ni ki ti A', ki = ki, ti=ti, ni=ni)
        return S 
    
    def AvgMidSelecting() : 
        pass 
        
    def ParMPDA() : 
        M = torch.zeros(hi, wi, dtype=torch.double)
        M = F.pad (M, (((fw-1)//2), ((fw-1)//2), ((fh-1)//2), ((fh-1)//2)), mode = 'constant', value=-0.5)
        M = torch.stack([ M[i:i+hi,j:j+wi] for i in range (fh) for j in range(fw)])
        M = einops.repeat(M, '(fh fw) h w -> fh fw (kk h s1 w s2)', fh = fh, fw = fw, kk = nt//(hi * ki * wi * ki), s1 = ki, s2 = ki)
        #U = U * M
        return M
    def ParMPDM() : 
        M = torch.ones(hi, wi, dtype=torch.double)
        M = F.pad (M, (((fw-1)//2), ((fw-1)//2), ((fh-1)//2), ((fh-1)//2)), mode = 'constant', value=0)
        M = torch.stack([ M[i:i+hi,j:j+wi] for i in range (fh) for j in range(fw)])
        M = einops.repeat(M, '(fh fw) h w -> fh fw (kk h s1 w s2)', fh = fh, fw = fw, kk = nt//(hi * ki * wi * ki), s1 = ki, s2 = ki)
        #U = U * M
        return M

    def PoolSelecting() : 
        S = torch.eye (ki*ti, ki *ti, dtype=torch.double) / (hi * wi)
        S = einops.repeat (S, 's1 s2 -> s1 (s2 ki)', ki = ki)
        S = F.pad (S, (0, ni * nt - S.shape[1]), mode = 'constant', value = 0)
        S = einops.rearrange (S, 's1 (ni X) -> ni s1 X', ni = ni)
        return S 

    def Downsamp (A) :
        # C = torch.zeros((ni,nt), dtype=torch.double)
        C = np.full( (ni) ,Empty(), dtype = object)
        S = DownSelecting()
        for i1 in range (ki) : # ky
            for i2 in range (ti) : 
                i3 = ((ki*i2 + i1)%(2*ko))//2 # Small y
                i4 = (ki*i2 + i1)%2 # Large x
                i5 = (ki*i2 + i1)//(2*ko) # Large y
                # i6 = (ki*i2 + i1) % cint(ci // ni)
                i7 = (ki*i2 + i1) // cint(ci // ni // ki )
                i8 = (ki*i2 + i1) // cint(co // no // ki )
                # C[i8] = C[i8] + roll (A[i7, :] * S[i7, i1, i2, :], - ((ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4) + (i8-i7) * nt) )
                C[i8] = C[i8] + roll (A[i7] * S[i7, i1, i2, :], - ((ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4) + (i8-i7) * nt) )
        for i in range(no) : 
            C[i] = roll (C[i], ko*ko*ho*wo*ti//8) # Centering
        # Centering is not working for multi ciphertext 
            for j in range(fint(np.log2(po))) : # Duplicating
                C[i] = C[i] + roll (C[i], pow(2, j) * (nt // po))
        return C

    def AvgPool(A) :
        # TODO: No multi ciphertext support
        #B = torch.zeros((no, nt), dtype=torch.double)
        #C = torch.zeros((no, nt), dtype=torch.double)
        B = np.full( (no) ,Empty(), dtype = object)
        C = np.full( (no) ,Empty(), dtype = object)
        S = PoolSelecting()
        for ii in range ( ni ) : 
            #B[ii, :] = A[ii, :]
            B[ii] = A[ii]
            for j in range (fint (np.log2(wi))) : 
                #B[ii, :] = B[ii,:] + roll (B[ii,:], - (pow(2, j) * ki))
                B[ii] = B[ii] + roll (B[ii], - (pow(2, j) * ki))
            for j in range (fint (np.log2(hi))) : 
                #B[ii, :] = B[ii, :] + roll (B[ii, :], - (pow(2, j) * ki * ki * wi))
                B[ii] = B[ii] + roll (B[ii], - (pow(2, j) * ki * ki * wi))
        for i1 in range(ki) :
            for i2 in range(ti) :
                i7 = (ki*i2 + i1) // (nt // (hi * wi))
                i8 = (ki*i2 + i1) // (nt // (ho * wo))
#                 i7 = (ki*i2 + i1) // cint(ci // ni // ki)
#                 i8 = (ki*i2 + i1) // cint(co // no // ki)
                #C[i8, :] = C[i8, :] + roll (B[i7, :], - (ki*ki*hi*wi*i2 + ki*wi * i1 - ki * (ki*i2 + i1) + (i8-i7)*nt)) *  S[ i7, ki* i2 + i1, :]
                C[i8] = C[i8] + roll (B[i7], - (ki*ki*hi*wi*i2 + ki*wi * i1 - ki * (ki*i2 + i1) + (i8-i7)*nt)) *  S[ i7, ki* i2 + i1, :]
        return C
    
    def AvgMidPool (A) :
        #B = torch.zeros((no, nt), dtype=torch.double)
        B = np.full( (ni) ,Empty(), dtype = object)

        if fw == 2 :
            for ii in range ( ni ) : 
                #B[ii, :] = A[ii, :]
                B[ii] = A[ii]
                for j in range (fint (np.log2(fw))) : 
                    #B[ii, :] = B[ii,:] + roll (B[ii,:], - (pow(2, j) * ki))
                    B[ii] = B[ii] + roll (B[ii], - (pow(2, j) * ki))
                for j in range (fint (np.log2(fh))) : 
                    #B[ii, :] = B[ii, :] + roll (B[ii, :], - (pow(2, j) * ki * ki * wi))
                    B[ii] = B[ii] + roll (B[ii], - (pow(2, j) * ki * ki * wi))
        else :
            M = ParMPDM()

            for ii in range(ni) : 
                for i1 in range(fh) :
                    for i2 in range(fw) : 
                        B[ii] = B[ii] + roll (A[ii], -(ki * ki * wi * (i1 - (fh-1)//2 ) + ki * (i2 - (fw-1)//2) ))  * M[i1, i2]

        
        #C = torch.zeros(no, nt, dtype=torch.double)
        C = np.full( (no) ,Empty(), dtype = object)
        S = DownSelecting()
        for i1 in range (ki) :
            for i2 in range (ti) :
                i3 = ((ki*i2 + i1)%(2*ko))//2
                i4 = (ki*i2 + i1)%2
                i5 = (ki*i2 + i1)//(2*ko)
                i7 = (ki*i2 + i1) // (nt // (hi * wi))
                i8 = (ki*i2 + i1) // (nt // (ho * wo))
                #C[i8, :] = C[i8, :] + roll (B[i7, :] * (S[i7, i1, i2, :]/ (fh * fw) ), - (ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4 + (i8-i7)*nt)) 
                C[i8] = C[i8] + roll (B[i7] * (S[i7, i1, i2, :]/ (fh * fw) ), - (ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4 + (i8-i7)*nt)) 
        # print(no)        
        for j in range(fint(np.log2(po))) :
            C[0] = C[0] + roll (C[0], pow(2, j) * (nt // po))
            # Temporary C[0]
            #C = C + roll (C, pow(2, j) * (nt // po))
        # print(C)
        # print(C.shape)
        return C
    
    def ConcatSelecting () : 
        tt = min (co * wo *ho , nt)
        M = torch.ones ((tt - (ci * wi * hi) % tt), dtype = torch.double)
        FF = F.pad (M, (0, (ci * wi * hi) %tt), mode = 'constant', value = 0)
        BB = 1 - FF
        FF = F.pad (FF, (0, nt//po - tt), mode = 'constant', value = 0)
        BB = F.pad (BB, (0, nt//po - tt), mode = 'constant', value = 0)
        FF = einops.repeat (FF, 'a -> (po a)', po = po)
        BB = einops.repeat (BB, 'a -> (po a)', po = po)
        
        return FF, BB
    
    def Concat (A, B):
        if ((ci * wi * hi) %nt) == 0 :
            return np.concatenate ((A, B))
        
        C = np.full((ni), Empty(), dtype=object)
        D = np.full((no), Empty(), dtype=object)
        FF, BB = ConcatSelecting()
        tt = min (co * wo *ho , nt)
        for i in range (ni) : 
            C[i] = roll (B[i], (ci * wi * hi) %tt)
        for i in range (ni-1) : 
            D[i] = A[i]
        
        first = [A[ni-1]] + [C[i]  for i in range(ni)]
        second = [C[i] for i in range(ni)]
        for i in range (ni) : 
            D[ni-1 +i] = first [i] * FF + second[i] * BB
        if ni != no : 
            D[no-1] = first[ni]
            
        return D
        
    
    def MaxPool (A) :
#         tmp = torch.zeros ( (ni, fh, fw, nt), dtype=torch.double)
#         for ii in range(ni) : 
#             for i1 in range(fh) :
#                 for i2 in range(fw) : 
#                     tmp[ii, i1, i2] = roll (A[ii, :], -(ki * ki * wi * i1   + ki * i2  ))
                
#         reduces = [tmp[:, i1, i2, :] for i1 in range(fh) for i2 in range(fw) ]
#         while len(reduces) > 1 :  
#             M = torch.zeros ((ni, nt) , dtype = torch.double)
#             for ii in range(ni) :
#                  M[ii, :] = maximum(reduces[-1][ii, :], reduces[-2] [ii, :])
#             reduces = [M] + reduces[:-2]
#         A = reduces[0]
        B = np.full((ni), Empty(), dtype=object)
        C = np.full((no), Empty(), dtype=object)
        for ii in range ( ni ) : 
            B[ii] = A[ii]
            for j in range (fint (np.log2(fw))) : 
                B[ii] = maximum(B[ii], roll (B[ii], - (pow(2, j) * ki)))
            for j in range (fint (np.log2(fh))) : 
                B[ii] = maximum(B[ii], roll (B[ii], - (pow(2, j) * ki * ki * wi)))
        
        # S = DownSelecting().cuda()
        S = DownSelecting()
        for i1 in range (ki) :
            for i2 in range (ti) :
                i3 = ((ki*i2 + i1)%(2*ko))//2
                i4 = (ki*i2 + i1)%2
                i5 = (ki*i2 + i1)//(2*ko)
                i7 = (ki*i2 + i1) // (nt // (hi * wi))
                i8 = (ki*i2 + i1) // (nt // (ho * wo))
                C[i8] = C[i8] + roll (B[i7] * S[i7, i1, i2, :], - (ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4 + (i8-i7)*nt)) 
                
                
#         for j in range(fint(np.log2(po))) :
#             C[0] = C[0] + roll (C[0], pow(2, j) * (nt // po))
            
        for ii in range(no) :
            for j in range(fint(np.log2(po))) :
                C[ii] = C[ii] + roll (C[ii], pow(2, j) * (nt // po))
        return C
    def MaxPoolPad (A) :
#         tmp = torch.zeros ( (ni, fh, fw, nt), dtype=torch.double)
#         for ii in range(ni) : 
#             for i1 in range(fh) :
#                 for i2 in range(fw) : 
#                     tmp[ii, i1, i2] = roll (A[ii, :], -(ki * ki * wi * i1   + ki * i2  ))
                
#         reduces = [tmp[:, i1, i2, :] for i1 in range(fh) for i2 in range(fw) ]
#         while len(reduces) > 1 :  
#             M = torch.zeros ((ni, nt) , dtype = torch.double)
#             for ii in range(ni) :
#                  M[ii, :] = maximum(reduces[-1][ii, :], reduces[-2] [ii, :])
#             reduces = [M] + reduces[:-2]
#         A = reduces[0]
        B = np.full((ni), Empty(), dtype=object)
        C = np.full((no), Empty(), dtype=object)
        M = ParMPDM()
        D = ParMPDA()
#         for ii in range ( ni ) : 
#             B[ii] = A[ii]
#             for j in range (fw) : 
#                 B[ii] = maximum(B[ii], roll (B[ii], - (j - (fw // 2)* ki)) * M[0, j] + D[0, j])
#             for j in range (fh) : 
#                 B[ii] = maximum(B[ii], roll (B[ii], - (j - (fh // 2) * ki * ki * wi))* M[j, 0] + D[j, 0])
        for ii in range ( ni ) : 
            B[ii] = A[ii]
            for j in range (fw) : 
                B[ii] = maximum(B[ii], roll (B[ii], - (j - (fw-1) // 2)* ki)  * M[1, j ] + D[1,j] )
            for j in range (fh) : 
                B[ii] = maximum(B[ii], roll (B[ii], - (j - (fh-1) // 2) * ki * ki * wi) * M[j, 1] + D[j,1]  )
        
        S = DownSelecting()
        for i1 in range (ki) :
            for i2 in range (ti) :
                i3 = ((ki*i2 + i1)%(2*ko))//2
                i4 = (ki*i2 + i1)%2
                i5 = (ki*i2 + i1)//(2*ko)
                i7 = (ki*i2 + i1) // (nt // (hi * wi))
                i8 = (ki*i2 + i1) // (nt // (ho * wo))
                C[i8] = C[i8] + roll (B[i7] * S[i7, i1, i2, :], - (ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4 + (i8-i7)*nt)) 
                
                
#         for j in range(fint(np.log2(po))) :
#             C[0] = C[0] + roll (C[0], pow(2, j) * (nt // po))
            
        for ii in range(no) :
            for j in range(fint(np.log2(po))) :
                C[ii] = C[ii] + roll (C[ii], pow(2, j) * (nt // po))
        return C

    def DwConvBN (A, U, G, H) :
        # D = torch.zeros ( (nt) , dtype=torch.double)
        # D = torch.zeros ( (no, nt) , dtype=torch.double)
        # tmp = torch.zeros ( (fh, fw, nt), dtype=torch.double)
        D = np.full( (no) ,Empty(), dtype = object)
        
        ## Need to use ParBnConst 
        ## Need to implement DwMultWgt
        tmp = np.empty((ni, fh, fw),  dtype = object)
        U = DwMultWgt(U)
        # print(U[0,0,0,0:64])
        P = ParInBNConst(G)
        for ii in range(ni) : 
            for i1 in range(fh) :
                for i2 in range(fw) : 
                    tmp[ii, i1, i2] = roll (A[ii], -(ki * ki * wi * (i1 - (fh-1)//2 ) + ki * (i2 - (fw-1)//2) ))
        # B = np.full( (ni) ,hc.Empty(), dtype = object)
        B = torch.zeros ( (nt) , dtype=torch.double)
        for ii in range(ni) :
            for i1 in range(fh) :
                for i2 in range(fw) :
                    B =  B + (tmp[ii,i1,i2] * U[ii, i1, i2, :])
        C = np.full( (no) ,Empty(), dtype = object)
        # S = DownSelecting().cuda()
        S = DownSelecting()
        for i1 in range (ki) :
            for i2 in range (ti) :
                i3 = ((ki*i2 + i1)%(s*ko))//s
                i4 = (ki*i2 + i1)%s
                i5 = (ki*i2 + i1)//(s*ko)
                i7 = (ki*i2 + i1) // cint(ci // ni)
                i8 = (ki*i2 + i1) // cint(co // no)
                C[i8] = C[i8] + roll (B * (S[i7, i1, i2, :] * P[i8] ), - (ki*ki*hi*wi * (i2-i5) + ki*wi * (i1-i3) - ki * i4 + (i8-i7)*nt)) 
                
        for ii in range(no) :
            for j in range(fint(np.log2(po))) :
                C[0] = C[ii] + roll (C[ii], pow(2, j) * (nt // po))
            C[ii] = C[ii] + ParBNConst(H)[ii, :]/bb
        return C
    
    def MultParBN(A, G, H) : 
        D = np.full( (no) ,Empty(), dtype = object)
        for ii in range(no) :
            D[ii] = A[ii] * ParBNConst(G)[ii, :]
            D[ii] = D[ii] + ParBNConst(H)[ii, :]/bb
        return D
 
    def MultParConv (A, U, bias) :
        
        # D = torch.zeros ( (nt) , dtype=torch.double)
        #D = torch.zeros ( (no, nt) , dtype=torch.double)
        D = np.full( (no) ,Empty(), dtype = object)
        # tmp = torch.zeros ( (fh, fw, nt), dtype=torch.double)
        tmp = np.empty((ni, fh, fw),  dtype = object)
        # tmp = np.full( (ni, fh, fw) ,hc.Empty(), dtype = object)
        U = ParMultWgt(U)
        S = Selecting()
        
        for ii in range(ni) : 
            for i1 in range(fh) :
                for i2 in range(fw) : 
                    tmp[ii, i1, i2] = roll (A[ii], -(ki * ki * wi * (i1 - (fh-1)//2 ) + ki * (i2 - (fw-1)//2) ))
        # T = roll(A, 1)
        
        for i3 in range(q) :
            B = torch.zeros ( (nt) , dtype=torch.double)
            for ii in range(ni) :
                # B = np.full((nt), hc.Empty(), dtype = object)
    #             B = hc.Empty()
                for i1 in range(fh) :
                    for i2 in range(fw) :
                        B =  B + (tmp[ii,i1,i2] * U[ii, i3, i1, i2, :])
                # T = roll(tmp, 1)
                # T = roll(U, 1)
            C = SumSlots(B, ki, 1)
            C = SumSlots(C, ki, ki*wi)
            C = SumSlots(C, ti, ki*ki*hi*wi)
            for i4 in range(min (pi, co-pi*i3)) :
                i = pi*i3 + i4
                i6 = i %  (nt // (hi * wi)) # tiled to 
                i8 = i // (nt // (hi * wi)) # out no
                D[i8] = D[i8] + roll (C, ((i6//(ko *ko))*ko *ko *ho * wo - (nt //pi) * (i6 % pi) +  ( (i6 % (ko *ko)) // ko) * ko * wo + (i6 %ko))- i8*nt) * (S[i8, i])

        for ii in range(no) :
            for j in range(fint(np.log2(po))) :
                D[ii] = D[ii] + roll (D[ii], pow(2, j) * (nt // po))
            D[ii] = D[ii] + ParBNConst(bias)[ii, :]/bb
        return D

    def MultParConvBN (A, U, G, H) :
        # D = torch.zeros ( (nt) , dtype=torch.double)
        #D = torch.zeros ( (no, nt) , dtype=torch.double)
        D = np.full( (no) ,Empty(), dtype = object)
        # tmp = torch.zeros ( (fh, fw, nt), dtype=torch.double)
        tmp = np.empty((ni, fh, fw),  dtype = object)
        # tmp = np.full( (ni, fh, fw) ,hc.Empty(), dtype = object)
        U = ParMultWgt(U)
        S = Selecting()
        # U = ParMultWgt(U.cuda())
        # S = Selecting().cuda()
        P = ParBNConst(G)
        for ii in range(ni) : 
            for i1 in range(fh) :
                for i2 in range(fw) : 
                    tmp[ii, i1, i2] = roll (A[ii], -(ki * ki * wi * (i1 - (fh-1)//2 ) + ki * (i2 - (fw-1)//2) ))
        # T = roll(A, 1)
        
        for i3 in range(q) :
            B = torch.zeros ( (nt) , dtype=torch.double)
            for ii in range(ni) :
                # B = np.full((nt), hc.Empty(), dtype = object)
    #             B = hc.Empty()
                for i1 in range(fh) :
                    for i2 in range(fw) :
                        B =  B + (tmp[ii,i1,i2] * U[ii, i3, i1, i2, :])
                # T = roll(tmp, 1)
                # T = roll(U, 1)
            C = SumSlots(B, ki, 1)
            C = SumSlots(C, ki, ki*wi)
            C = SumSlots(C, ti, ki*ki*hi*wi)
            for i4 in range(min (pi, co-pi*i3)) :
                i = pi*i3 + i4
                i6 = i %  (ko * ko *nt // (hi * wi * ki *ki)) # tiled to 
                i8 = i // (ko * ko * nt // (hi * wi * ki * ki)) # out no
                D[i8] = D[i8] + roll (C, ((i6//(ko *ko))*ko *ko *ho * wo - (nt //pi) * (i6 % pi) +  ( (i6 % (ko *ko)) // ko) * ko * wo + (i6 %ko))- i8*nt) * (S[i8, i]* P[i8])

        for ii in range(no) :
            for j in range(fint(np.log2(po))) :
                D[ii] = D[ii] + roll (D[ii], pow(2, j) * (nt // po))
            D[ii] = D[ii] + ParBNConst(H)[ii, :]/bb
        return D
    
    
    # return {"MPP" : MultParPack, "MPCB" : MultParConvBN, "DS" : Downsamp, "AP" : AvgPool, "MP" : MaxPool, "MA": AvgMidPool , "OP": OutPack, "PBN" : ParBNConst}
    return {"MPP" : MultParPack, "CC": Concat, "MPD" : MaxPoolPad,"MPCB" : MultParConvBN, "BN" :MultParBN,  "MPC": MultParConv, "DW" : DwConvBN, "DS" : Downsamp, "AP" : AvgPool, "MP" : MaxPool, "MA": AvgMidPool , "OP": OutPack}



def abstractBN (N):
    T = N.weight
    V = N.running_var
    M = N.running_mean
    I = N.bias
    e = N.eps 
    G = T / torch.sqrt(V + e)
    H = I - G * M
    return G, H

def Linear (A, U, B, nt) :
    R = np.full( (1) ,Empty(), dtype = object)
    #R = torch.zeros ((1, nt), dtype=torch.double)
    outdim = U.shape[0] #10
    indim = U.shape[1] #64
    S = torch.ones(indim)
    #
    S = F.pad(S, (0, nt- S.shape[0]), mode= 'constant', value=0)
    #A[0, :] = S * A[0, :]
    A[0] = S * A[0]
    #
    #A[0, :] = A[0, :] + roll (A[0, :], indim)
    A[0] = A[0] + roll (A[0], indim)
    it = (indim + outdim - 1) // outdim # 7
    U = torch.stack([ torch.roll (U[i,:], -i) for i in range (outdim) ])
    U = F.pad(U, (0, it*outdim - indim) , mode = "constant", value = 0 )
    U = einops.rearrange(U, "i1 (i2 i3) -> i3 (i2 i1)", i1 = outdim, i2 = it, i3 = outdim)
    U = F.pad(U, (0, nt- U.shape[1]), mode = 'constant', value = 0)
    for i in range (outdim) :
        #R[0, :] = R[0,:] + roll(A[0, :], -i) * U[i, :]
        R[0] = R[0] + roll(A[0], -i) * U[i, :]
    for j in range(cint(np.log2(it))) :
        #R[0, :] = R[0, :] + roll (R, -pow(2, j) * outdim)
        R[0] = R[0] + roll (R[0], -pow(2, j) * outdim)
    B = F.pad(B, (0, nt-B.shape[0]) , mode = "constant", value = 0 )
    #R[0, :] = R[0, :]+B
    R[0] = R[0]+B
    return R

def BN (A, G, H, nt):
    R = np.full( (1) ,Empty(), dtype = object)
    #R = torch.zeros((1, nt), dtype = torch.double)
    G = F.pad(G, (0, nt - G.shape[0]) , mode = "constant", value = 0 )
    H = F.pad(H, (0, nt - H.shape[0]) , mode = "constant", value = 0 )
    #R[0, :] = A[0, :] * G + H
    R[0] = A[0] * G + H
    return R

def Reshape (W, shape) : 
    print (W.shape )
    ko = shape["ko"]
    wo = shape["wo"]
    ho = shape["ho"]
    to = shape["to"]
    
    W = einops.rearrange(W, "outdim (to ko1 ko2 ho wo) -> outdim (to ho ko1 wo ko2)", outdim = W.shape[0], to = to, ko2 = ko, ko1 = ko, wo = wo, ho = ho)
    return W


def InferShapes(shapes) : 
    shapes["ho"] = shapes["hi"]//shapes["s"]
    shapes["wo"] = shapes["wi"]//shapes["s"]
    shapes["ko"] = shapes["s"] * shapes["ki"]
    shapes["ti"] = cint(shapes["ci"] / (shapes["ki"]*shapes["ki"]))
    shapes["to"] = cint(shapes["co"] / (shapes["ko"]*shapes["ko"]))
    shapes["ni"] = cint ((shapes["ki"] * shapes["ki"] * shapes["hi"] * shapes["wi"] * shapes["ti"])/shapes["nt"])
    shapes["no"] = cint ((shapes["ko"] * shapes["ko"] * shapes["ho"] * shapes["wo"] * shapes["to"])/shapes["nt"])
    shapes["pi"] = pow(2, fint(np.log2((shapes["nt"] )/ (shapes["ki"] * shapes["ki"] * shapes["hi"] * shapes["wi"] * shapes["ti"]))))
    shapes["po"] = pow(2, fint(np.log2((shapes["nt"] )/ (shapes["ko"] * shapes["ko"] * shapes["ho"] * shapes["wo"] * shapes["to"]))))
    if shapes["pi"] < 1 : 
        shapes["pi"] = 1
    if shapes["po"] < 1 : 
        shapes["po"] = 1
    shapes["q"] =  cint(shapes["co"]/ shapes["pi"])
    return shapes
    
def CascadeConcat (shapes1, shapes2) : 
    shapes = shapes1.copy()
    if (shapes1["co"] != shapes2["co"]) or (shapes1["co"] % (shapes["ko"] * shapes["ko"]) != 0):  
        print ("concat does not support this shape")
        return 0
    else :
        shapes["fh"] =  1
        shapes["fw"] =  1 
        shapes["s"] = 1
        
        shapes["ci"] = shapes1["co"]
        shapes["co"] = shapes["ci"] *2
        
        shapes["hi"] = shapes["ho"]
        shapes["wi"] = shapes["wo"]
        shapes["ki"] = shapes["ko"]
        
        return InferShapes(shapes)
        
        

def CascadeConv (shapes, Conv) : 
    # Conv Characteristics
    shapes = shapes.copy()
    shapes["fh"] = Conv.kernel_size[1] 
    shapes["fw"] = Conv.kernel_size[0] 
    shapes["s"] = Conv.stride[0] 
    
    # Channel Characteristics
    shapes["ci"] = Conv.in_channels 
    shapes["co"] = Conv.out_channels 

    # Input Characteristics (Cascaded)
    
    shapes["hi"] = shapes["ho"]
    shapes["wi"] = shapes["wo"]
    shapes["ki"] = shapes["ko"]
    
    return InferShapes(shapes)
    
def CascadeMax (shapes, Max) :
    # Conv Characteristics
    shapes = shapes.copy()
    if type(Max.kernel_size) is int :
        shapes["fh"] = Max.kernel_size
        shapes["fw"] = Max.kernel_size
    else:
        shapes["fh"] = Max.kernel_size[1]
        shapes["fw"] = Max.kernel_size[0]
    if type(Max.stride) is int :
        shapes["s"] = Max.stride
    else:
        shapes["s"] = Max.stride[0]
    
    # Channel Characteristics
    co = shapes["co"]
    shapes["ci"] = co 
    shapes["co"] = co 

    # Input Characteristics (Cascaded)
    
    shapes["hi"] = shapes["ho"]
    shapes["wi"] = shapes["wo"]
    shapes["ki"] = shapes["ko"]
    
    return InferShapes(shapes)
    

def CascadeDS (shapes) :
    # Conv Characteristics
    shapes = shapes.copy()
    shapes["fh"] =  1
    shapes["fw"] =  1 
    shapes["s"] = 2 
    
    # Channel Characteristics
    co = shapes["co"]
    shapes["ci"] = co 
    shapes["co"] = co *2

    # Input Characteristics (Cascaded)
    
    shapes["hi"] = shapes["ho"]
    shapes["wi"] = shapes["wo"]
    shapes["ki"] = shapes["ko"]
    
    return InferShapes(shapes)
    
def CascadePool (shapes) : 
    # Conv Characteristics
    shapes = shapes.copy()
    shapes["fh"] =  1
    shapes["fw"] =  1 
    shapes["s"] = 1
    
    # Channel Characteristics
    co = shapes["co"]
    shapes["ci"] = co 
    shapes["co"] = co 

    # Input Characteristics (Cascaded)

    shapes["hi"] = shapes["ho"]
    shapes["wi"] = shapes["wo"]
    shapes["ki"] = shapes["ko"]
    
    return InferShapes(shapes)






# In[19]:


