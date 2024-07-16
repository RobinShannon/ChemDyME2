import numpy as np
from ase.io import read

TS_hess = np.load('GlyoxalGeoms/NN_TS.npy')
post_comp = np.load('GlyoxalGeoms/ProductProjection.npy')
HCOCO_hess = np.load('GlyoxalGeoms/HCOCO_linear.npy')
water_hess = np.load('GlyoxalGeoms/water.npy')
#imag = np.asarray([-0.02,0.01,0,0.01,0,0,0,0,0,0.99,-0.11,0,0.03,-0.01,0,-0.01,-0.02,0,-0.08,0.01,0,-0.08,0.01,0])
imag = TS_hess[:,-1]
comp_arr = []
comp_sum = 0
for i in range(0,24):
    vec = post_comp[:,i]
    proj = abs(np.dot(imag,vec)/np.sqrt(np.dot(vec,vec)))
    comp_arr.append(proj)
    comp_sum += proj

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for j in range(0,24):
    imag = post_comp[:,j]
    imag_h = imag[[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]]
    imag_w = imag[[9, 10, 11, 18, 19, 20, 21, 22, 23]]

    for i in range(0, 15):
        vec = HCOCO_hess[:, i]
        proj = abs(np.dot(imag_h, vec)/np.sqrt(np.dot(vec,vec))) * comp_arr[j]
        h_arr.append(proj)
        if i > 2:
            h_sum += proj
        else:
            trans_sum += proj

    for i in range(0, 9):
        vec = water_hess[:, i]
        proj = abs(np.dot(imag_w, vec)/np.sqrt(np.dot(vec,vec))) * comp_arr[j]
        w_arr.append(proj)
        if i > 2:
            w_sum += proj
        else:
            trans_sum += proj

total = w_sum+h_sum+trans_sum


imag = -1*TS_hess[:,-1]
imag_h = imag[[0,1,2,3,4,5,6,7,8,12,13,14,15,16,17]]
imag_w = imag[[9,10,11,18,19,20,21,22,23]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,15):
    vec = HCOCO_hess[:,i]
    proj = abs(np.dot(imag_h,vec) / np.sqrt(np.dot(vec,vec)))
    h_arr.append(proj)
    if i >2:
        h_sum += proj
    else:
        trans_sum += proj

for i in range(0,9):
    vec = water_hess[:,i]
    proj = abs(np.dot(imag_w,vec) / np.sqrt(np.dot(vec,vec)))
    w_arr.append(proj)
    if i >2:
        w_sum += proj
    else:
        proj_h = abs(np.dot(imag_w[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_w[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        w_sum += 0.5 *proj_oh
        trans_sum += 0.5 * proj_oh
        h_sum += 0.5 * proj_h
        w_sum += 0.5 * proj_h

total = w_sum+h_sum+trans_sum
#print("water proportion = " + str(w_sum/total))
#print("hcoco proportion = " + str(h_sum/total))
#print("trans proportion = " + str(trans_sum / total ))
#print(" Gly done")


TS_hess = np.load('MethylFormate/TS.npy')
HCOCO_hess = np.load('MethylFormate/co.npy')
water_hess = np.load('MethylFormate/water.npy')

imag = TS_hess[:,-1]

imag_h = imag[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,21,22,23]]
imag_w = imag[[18,19,20,24,25,26,27,28,29]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,21):
    vec = HCOCO_hess[:,i]
    proj = abs(np.dot(imag_h,vec) / np.dot(vec,vec))
    h_arr.append(proj)
    if i >2:
        h_sum += proj
    else:
        trans_sum += proj

for i in range(0,9):
    vec = water_hess[:,i]
    proj = abs(np.dot(imag_w,vec) / np.dot(vec,vec))
    w_arr.append(proj)
    if i > 2:
        w_sum += proj
    else:
        w_sum += proj

total = w_sum+h_sum+trans_sum
print("water proportion = " + str(w_sum/total))
print("MF proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print(" MF done")

TS_hess = np.load('H2O2/ts.npy')
HCOCO_hess = np.load('H2O2/ho2.npy')
water_hess = np.load('H2O2/water.npy')

imag = TS_hess[:,-1]

imag_h = imag[[0,1,2,3,4,5,6,7,8]]
imag_w = imag[[9,10,11,12,13,14,15,16,17]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,9):
    vec = HCOCO_hess[:,i]
    proj = abs(np.dot(imag_h,vec) / np.dot(imag_h,imag_h))
    h_arr.append(proj)
    if i >2:
        h_sum += proj
    else:
        proj_h = abs(np.dot(imag_h[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_h[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += proj_oh
        h_sum += proj_h

for i in range(0,9):
    vec = water_hess[:,i]
    proj = abs(np.dot(imag_w,vec) / np.dot(imag_h,imag_h))
    w_arr.append(proj)
    if i > 2:
        w_sum += proj
    else:
        proj_h = abs(np.dot(imag_w[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_w[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += 1 * proj_oh
        w_sum += 1 * proj_h

total = w_sum+h_sum+trans_sum


TS_hess = np.load('Formaldehyde/TS2.npy')
HCOCO_hess = np.load('Formaldehyde/Fomaldehyde2.npy').T
water_hess = np.load('Formaldehyde/water2.npy').T
imag = [0.28,0.17,0,0.65,0.02,0,0.35,0.52,0,-0.02,-0.06,0,-0.25,-0.09,0,-0.04,-0.07,0]
imag=np.asarray(imag)
imag = TS_hess[-1,:]

imag_h = imag[[0,1,2,3,4,5,6,7,8,9,10,11]]
imag_w = imag[[6,7,8,12,13,14,15,16,17]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,12):
    vec = HCOCO_hess[i,:]
    proj = abs(np.dot(imag_h,vec) / np.dot(imag_h,imag_h))
    h_arr.append(proj)
    if i >2:
        h_sum += proj
    else:
        trans_sum += proj


for i in range(0,9):
    vec = water_hess[:,i]
    proj = abs(np.dot(imag_w,vec) / np.dot(imag_w,imag_w))
    w_arr.append(proj)
    if i >2:
        w_sum += proj
    else:
        w_sum +=proj


total = w_sum+h_sum+trans_sum
print("water proportion = " + str(w_sum/total))
print("formaldehyde proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("Form OH done")

TS_hess = np.load('FormH/ts2.npy').T
HCO_hess = np.load('FormH/HCO2.npy').T
h2_hess = np.load('FormH/H22.npy').T

#imag = [-0.02,0.01,0,0.02,0,0,0.07,0,-0.01,0.58,0.36,0,-0.62,-0.38,0]
#imag=np.asarray(imag)
imag = TS_hess[:,-1]

imag_hco = imag[[0,1,2,3,4,5,6,7,8]]
imag_h = imag[[9,10,11,12,13,14]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,9):
    vec = HCO_hess[:,i]
    proj = abs(np.dot(imag_hco,vec) / np.dot(vec,vec))
    h_arr.append(proj)
    if i >2:
        h_sum += proj
    else:
        proj_h = abs(np.dot(imag_hco[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_hco[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += 1 * proj_oh
        h_sum += 1 * proj_h


for i in range(0,6):
    vec = h2_hess[:,i]
    proj = abs(np.dot(imag_h,vec) / np.dot(vec,vec))
    w_arr.append(proj)
    if i > 2:
        w_sum += proj
    else:
        proj_h = abs(np.dot(imag_h[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_h[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += 1 * proj_oh
        h_sum += 1 * proj_h


#total = w_sum+h_sum+trans_sum
#print("H2 proportion = " + str(w_sum/total))
#print("hco proportion = " + str(h_sum/total))
#print("trans proportion = " + str(trans_sum / total ))
#print("done")

TS_hess = np.load('FormH/tscorrected.npy')
comp_hess = np.load('FormH/prodcompcorrected.npy')


#imag = [-0.02,0.01,0,0.02,0,0,0.07,0,-0.01,0.58,0.36,0,-0.62,-0.38,0]
#imag=np.asarray(imag)
imag = TS_hess[:,-1]



h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(6,15):
    vec = comp_hess[:,i]
    proj = abs(np.dot(imag,vec) / np.dot(vec,vec))
    h_arr.append(proj)
    if i > 9:
        trans_sum += proj
    elif i >8:
        w_sum += proj
    else:
        h_sum += proj




#total = w_sum+h_sum+trans_sum
#print("H2 proportion = " + str(w_sum/total))
#print("hco proportion = " + str(h_sum/total))
#print("trans proportion = " + str(trans_sum / total ))
#print("done")

H21 = read('FormH/H2proj.xyz')
H22 = read('FormH/H2_NN.xyz')
H2Hess = np.load('FormH/H2corrected.npy')
H2_mu = 1/(1.953*1.953)
H2_freq = 3585.9
H2_disp = H22.get_positions() - H21.get_positions()
H2_disp = H2_disp.flatten()
H2inv = np.dot(np.linalg.inv(H2Hess),H2_disp)
H2sum = 0
for i in range(5,len(H2inv)):
    H2sum += H2_mu*0.5*(H2_freq*H2inv[i])**2

HCO21 = read('FormH/HCOproj.xyz')
HCO22 = read('FormH/HCO_NN.xyz')
HCO2Hess = np.load('FormH/HCOcorrected.npy')
HCO2_mu = [0,0,0,0,0,0,1 / (1.27*1.27), 1 /(1.13*1.13), 1/(1.34*1.34)]
HCO2_freq = [0,0,0,0,0,0,1107,2008,2745]
HCO2_disp = HCO22.get_positions() - HCO21.get_positions()
HCO2_disp = HCO2_disp.flatten()
HCO2inv = np.linalg.inv(HCO2Hess)*HCO2_disp

HCO2inv = np.dot(np.linalg.inv(HCO2Hess),HCO2_disp)
HCO2sum = 0
for i in range(6,len(HCO2inv)):
    HCO2sum += HCO2_mu[i]*0.5*(HCO2_freq[i]*HCO2inv[i])**2

#print(str((HCO2sum/ (HCO2sum+H2sum))))
#print(str((H2sum/ (HCO2sum+H2sum))))

TS_hess = np.load('FormH/tscorrected.npy')
HCO_hess = np.load('FormH/Formaldehydecorrected.npy')
h2_hess = np.load('FormH/H22.npy')

#imag = [-0.02,0.01,0,0.02,0,0,0.07,0,-0.01,0.58,0.36,0,-0.62,-0.38,0]
#imag=np.asarray(imag)
imag = TS_hess[:,-1]

imag_hco = imag[[0,1,2,3,4,5,6,7,8,9,10,11]]
imag_h = imag[[9,10,11,12,13,14]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,12):
    vec = HCO_hess[:,i]
    proj = abs(np.dot(imag_hco,vec) / np.dot(vec,vec))
    h_arr.append(proj)
    if i >2:
        h_sum += proj
    else:
        trans_sum += proj


for i in range(0,6):
    vec = h2_hess[:,i]
    proj = abs(np.dot(imag_h,vec) / np.dot(vec,vec))
    w_arr.append(proj)
    if i > 2:
        w_sum += proj
    else:
        w_sum += proj


total = w_sum+h_sum+trans_sum
print("H2 proportion = " + str(w_sum/total))
print("hco proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("done")
