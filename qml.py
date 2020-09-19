import sys 
import numpy as np
#had to install pandas on python2 cuz thtat's what it uses?
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_encoding = "ISO-8859-1"
data = pd.read_csv("data.csv", encoding = file_encoding)
PCA = PCA(n_components = 2)

def build_smoking_pc(tobacco, smoking, smoking_freq):
    x = data[["household_tobacco", "household_smoking", "household_smoking_freq"]]
    x.append([tobacco, smoking, smoking_freq])
    x = StandardScaler().fit_transform(x)
    principalComponents = PCA.fit_transform(x)
    smoking_env = pd.DataFrame(data = principalComponents, columns = ['pc_smoke1', 'pc_smoke2'])

    return [smoking_env["pc_smoke1"][len(smoking_env)-1],
            smoking_env["pc_smoke2"][len(smoking_env)-1]]


def build_phys_pc(vig_work, mod_work, bike_walk, vig_play, mod_play):
    x = data[[ "vig_work_freq", "mod_work_freq", "bike_walk_freq","vig_play_freq","mod_play_freq" ]]
    x.append([vig_work, mod_work, bike_walk, vig_play, mod_play])

    x = StandardScaler().fit_transform(x)
    principalComponents = PCA.fit_transform(x)
    physical_activity = pd.DataFrame(data = principalComponents, columns = ['pc_phys1', 'pc_phys2'])

    return [physical_activity["pc_phys1"][len(physical_activity)-1],
        physical_activity["pc_phys2"][len(physical_activity)-1]]

def build_profile(args):
    profile = []
    #gender
    profile.append(int(args[1]))

    #################################

    #age
    if(int(args[2]) >= 0) & (int(args[2]) <= 39):
        #TODO: get data from 0-18 yr olds
        #young
        profile.append(1)
    else:
        profile.append(0)

    if (int(args[2]) >= 40) & (int(args[2]) <= 55):
        #middle
        profile.append(1)
    else:
        profile.append(0)

    #################################

    #race
    if int(args[3]) == 3:
        #white
        profile.append(1)
    else:
        profile.append(0)

    if int(args[3]) == 6:
        #asian
        profile.append(1)
    else:
        profile.append(0)

    if int(args[3]) == 4:
        #black
        profile.append(1)
    else:
        profile.append(0)

    if int(args[3]) == 1:
        #mexam
        profile.append(1)
    else:
        profile.append(0)

    #################################

    #obese
    profile.append(int(args[4]))

    #avg_salt_table
    profile.append(int(args[5]))

    #everyday_cig
    if int(args[6]) == 1:
        profile.append(1)
    else:
        profile.append(0)

    #somedays_cig
    if int(args[6]) == 2:
        profile.append(1)
    else:
        profile.append(0)

    #smoking_pcs
    smoking_pcs = build_smoking_pc(int(args[7]), int(args[8]), int(args[9]))
    profile.append(smoking_pcs[0])
    profile.append(smoking_pcs[1])

    #phys_activity pcs
    phys_pcs = build_phys_pc(int(args[10]), int(args[11]), int(args[12]), int(args[13]), int(args[14]))
    profile.append(phys_pcs[0])
    profile.append(phys_pcs[1])


    return profile

profile = build_profile(sys.argv)
# print(profile)

systolic_coef = [-3.22894826, -16.43366318, -14.27867986,  -2.1707895,  -1.60189382, 2.05835427,  -1.2727281,    2.82616044,  2.08099019,   0.13132413, 1.54374152,   1.7508434,   -0.34450985,  -0.27499765,  -0.37484246]
systolic_intercept = 132.88965159

diastolic_coef = [-1.94056473,  -1.31152627, -1.14557152,  0.26182935,  3.03297809,  0.97890963,  -0.18695312, 2.70276957,  2.42145287, -1.22586465, 1.2614506,   1.49943734,   0.03091051,  -1.31753543, -0.24487827]
diastolic_intercept = 75.8483776 #TODO FIX 65 -> 70

#og-values
#smoking pc1 for both systolic & diastolic -> sys - 0.7508434; dia 0.49943734
#middle  sys -18.27867986
#avg_salt_table sys - 1.08099019, dia - 1.42145287
#pc_phys1 dia - 0.31753543
#sys young -11.43366318 dia young 5.31152627

def predict_systolic(coefficients, intercept, profile):
    return np.dot(coefficients, profile) + intercept

def predict_diastolic(coefficients, intercept, profile):
    return np.dot(coefficients, profile) + intercept

systolic_pressure = int(np.round(predict_systolic(systolic_coef, systolic_intercept, profile)))
diastolic_pressure = int(np.round(predict_diastolic(diastolic_coef, diastolic_intercept, profile)))
print( str(systolic_pressure) + "/" + str(diastolic_pressure))
