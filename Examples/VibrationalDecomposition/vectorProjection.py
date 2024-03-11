import numpy as np

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
print("water proportion = " + str(w_sum/total))
print("hcoco proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("done")


imag = TS_hess[:,-1]
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
print("water proportion = " + str(w_sum/total))
print("hcoco proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("done")


TS_hess = np.load('MethylFormate/NN_TS.npy')
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
        proj_h = abs(np.dot(imag_w[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_w[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += 0.5 * proj_oh
        w_sum += 0.5 * proj_oh
        w_sum += 0.5 * proj_h
        h_sum += 0.5 * proj_h

total = w_sum+h_sum+trans_sum
print("water proportion = " + str(w_sum/total))
print("hcoco proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("done")

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
        proj_h = abs(np.dot(imag_w[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_w[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += 0.5 * proj_oh
        w_sum += 0.5 * proj_oh
        w_sum += 0.5 * proj_h
        h_sum += 0.5 * proj_h

total = w_sum+h_sum+trans_sum
print("water proportion = " + str(w_sum/total))
print("hcoco proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("done")

TS_hess = np.load('Formaldehyde/ts.npy')
HCOCO_hess = np.load('Formaldehyde/HCO.npy')
water_hess = np.load('Formaldehyde/water.npy')

imag = TS_hess[:,-1]

imag_h = imag[[0,1,2,3,4,5,9,10,11]]
imag_w = imag[[6,7,8,12,13,14,15,16,17]]

h_sum=0
trans_sum = 0
w_sum = 0
h_arr = []
w_arr = []

for i in range(0,9):
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
        proj_h = abs(np.dot(imag_w[:3], vec[:3]) / np.sqrt(np.dot(vec[:3], vec[:3])))
        proj_oh = abs(np.dot(imag_w[3:], vec[3:]) / np.sqrt(np.dot(vec[3:], vec[3:])))
        sum = proj_h + proj_oh
        proj_h *= proj / sum
        proj_oh *= proj / sum
        trans_sum += 0.5 * proj_oh
        w_sum += 0.5 * proj_oh
        h_sum += 0.5 * proj_h
        w_sum += 0.5 * proj_h

total = w_sum+h_sum+trans_sum
print("water proportion = " + str(w_sum/total))
print("hcoco proportion = " + str(h_sum/total))
print("trans proportion = " + str(trans_sum / total ))
print("done")