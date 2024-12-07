import os
import copy
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib as mpl
from config import Config
from itertools import product
import matplotlib.pyplot as plt

sns.set(context='notebook', style='ticks', font_scale=1.5, font='sans-serif', rc={"lines.linewidth": 1.2})
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.despine(left=True, bottom=True)
bg_color = (0.88, 0.85, 0.95)
bg_color = (1, 1, 1)
bg_color = 'white'

plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
plt.rcParams["savefig.facecolor"] = bg_color
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
plt.rcParams.update({"savefig.format": 'png'})
cmap_name = 'Blues'

renaming_columns = {
        'Brain stem': 'BrainStem',
        'left pallidum': 'left pallidium',
        'right pallidum': 'right palladium',
        'left cerebellum exterior': 'left cerebellem exterior'
    }

parkTopFeatures = ['vol_mtg_sn_snc_leftcit168',
        'vol_right_pars_orbitalisdktregions',
        'vol_brain_stemdktregions',
        'vol_parietal_ldktlobes',
        'area_left_perirhinalmtl',
        'thk_mtg_sn_snc_leftcit168',
        'vol_mtg_sn_snc_leftdeep_cit168',
        'thk_mtg_sn_snr_leftcit168',
        'vol_referenceregion_rightcit168',
        'vol_mtg_vtr_pbp_leftcit168',
        'vol_bn_gp_gpe_rightcit168',
        'vol_mtg_sn_snc_rightcit168',
        'thk_mtg_vtr_pbp_rightcit168',
        'vol_right_rostral_middle_frontaldktregions',
        'thk_bn_gp_gpe_rightcit168',
        'thk_thm_eth_hn_leftcit168',
        'thk_right_parahippocampaldktregions',
        'vol_left_cerebellum_white_matterdktregions',
        'vol_cerebellar_vermal_lobules_i.vdktregions']

manualMaps = {'vol_parietal_ldktlobes': 'left superior parietal',
        'vol_bn_gp_gpe_rightcit168': 'right pallidium',
        'thk_bn_gp_gpe_rightcit168': 'right pallidium',
        'vol_cerebellar_vermal_lobules_i.vdktregions': 'cerebellar vermal lobules I-V'}

brainTemplate = "segmentedTemplate.nii.gz"

fullFeatList = pd.read_csv('labeledFeatureList.csv')
colToRegion = {}
for col in parkTopFeatures:
    for row in fullFeatList.itertuples():
        if col in manualMaps and manualMaps[col] == row.region_name:
            colToRegion[col] = row.region_index
        elif row.region_name.lower() in col.replace('_', ' ').replace('.', ' ').lower():
            colToRegion[col] = row.region_index

config = Config()
NUM_RUNS = config.num_runs

staticMap = pickle.load(open('/home/SECONDGRAM/data/staticMap.pkl', 'rb'))
featureMap = pickle.load(open('/home/SECONDGRAM/data/featureMap.pkl', 'rb'))
scaleFirst = pickle.load(open('/home/SECONDGRAM/data/scaleFirst.pkl', 'rb'))
scaleSecond = pickle.load(open('/home/SECONDGRAM/data/scaleSecond.pkl', 'rb'))

key = 'secondgram'
RUN_NUM = random.randint(1, NUM_RUNS)
NUM_SAMPLES = 3

realData = pickle.load(open('/home/SECONDGRAM/data/trainData.pkl', 'rb')) + pickle.load(open('/home/SECONDGRAM/data/valData.pkl', 'rb')) + pickle.load(open('/home/SECONDGRAM/data/testData.pkl', 'rb'))
generatedData = pickle.load(open(f'/home/SECONDGRAM/generations/generatedTrainData_{key}_{RUN_NUM}.pkl', 'rb')) + pickle.load(open(f'/home/SECONDGRAM/generations/generatedValData_{key}_{RUN_NUM}.pkl', 'rb')) + pickle.load(open(f'/home/SECONDGRAM/generations/generatedTestData_{key}_{RUN_NUM}.pkl', 'rb'))
allData = [(realData[i][0], realData[i][1], realData[i][2], generatedData[i]) for i in range(len(realData)) if realData[i][2] is not None]

allData = [(p[0], scaleFirst.inverse_transform(p[1].reshape(1, -1)).squeeze(), scaleSecond.inverse_transform(p[2].reshape(1, -1)).squeeze(), scaleSecond.inverse_transform(p[3].reshape(1, -1)).squeeze()) for p in allData]

park = [p for p in allData if p[0][staticMap['PARKINSONISM']] == 1]
park = random.sample(park, NUM_SAMPLES)
noPark = [p for p in allData if p[0][staticMap['PARKINSONISM']] == 0]
noPark = random.sample(noPark, NUM_SAMPLES)

minMaxFeatures = {f: (min([img[featureMap[f]] for p in park+noPark for img in p[1:]]), max([img[featureMap[f]] for p in park+noPark for img in p[1:]])) for f in parkTopFeatures}

templatedImage = nib.load(brainTemplate).get_fdata()
allIdx = fullFeatList.region_index.unique().tolist()
if not os.path.exists(f'/home/SECONDGRAM/temp/template_background.png'):
    os.makedirs('/home/SECONDGRAM/temp/', exist_ok=True)
    segmentedImage = templatedImage.copy()
    n_columns = 3
    num_rows = 2
    fig, axs = plt.subplots(nrows=num_rows, ncols=n_columns, figsize=(182 * n_columns / 18, 9 * num_rows))
    plt.subplots_adjust(hspace=0)
    axslist = axs.reshape(-1)
    for e, (slice_number, axis_number) in enumerate(product([60, 90], [0, 1, 2])):
        ax = axslist[e]
        if axis_number == 0:
            numpy_image = np.rot90(segmentedImage[slice_number, :, :])
        elif axis_number == 1:
            numpy_image = np.rot90(segmentedImage[:, slice_number, :])
        else:
            numpy_image = np.rot90(segmentedImage[:, :, slice_number])

        SEG_NUMPY = pd.DataFrame(numpy_image.astype(int))
        A = np.unique(SEG_NUMPY)
        ax.set_axis_off()
        replacement_dict = {idx: 1 for idx in allIdx}
        for idx in A:
            replacement_dict[idx] = replacement_dict.get(idx, np.nan)
        V = SEG_NUMPY.copy().replace(replacement_dict).values
        cmap = copy.copy(mpl.cm.get_cmap('Greys'))
        cmap.set_bad("white")
        ax.set_axis_off()
        sns.heatmap(V, cmap=cmap, mask=V == np.nan, ax=ax, center=0,
                        cbar=None, cbar_ax=None, rasterized=True)
    fig.tight_layout()
    plt.savefig(f'/home/SECONDGRAM/temp/template_background.png', dpi=200)
    plt.close()

    
for (k, d) in [('Parkinsonism', park), ('NoParkinsonism', noPark)]:
    for i in range(NUM_SAMPLES):
        for k2, img in [('First', d[i][1]), ('Real', d[i][2]), ('Generated', d[i][3])]:
            pdFeatures = {f: img[featureMap[f]] for f in parkTopFeatures}
            pdFeatures = {f: (pdFeatures[f] - minMaxFeatures[f][0]) / (minMaxFeatures[f][1] - minMaxFeatures[f][0]) for f in pdFeatures}
            indexMap = {colToRegion[f]: pdFeatures[f] for f in pdFeatures if f in colToRegion}
            segmentedImage = templatedImage.copy()
            n_columns = 3
            num_rows = 2
            fig, axs = plt.subplots(nrows=num_rows, ncols=n_columns, figsize=(182 * n_columns / 18, 9 * num_rows))
            plt.subplots_adjust(hspace=0)
            axslist = axs.reshape(-1)
            for e, (slice_number, axis_number) in enumerate(product([60, 90], [0, 1, 2])):
                ax = axslist[e]
                if axis_number == 0:
                    numpy_image = np.rot90(segmentedImage[slice_number, :, :])
                elif axis_number == 1:
                    numpy_image = np.rot90(segmentedImage[:, slice_number, :])
                else:
                    numpy_image = np.rot90(segmentedImage[:, :, slice_number])
                
                SEG_NUMPY = pd.DataFrame(numpy_image.astype(int))
                A = np.unique(SEG_NUMPY)
                ax.set_axis_off()
                replacement_dict = indexMap
                for idx in A:
                    replacement_dict[idx] = replacement_dict.get(idx, np.nan)
                V = SEG_NUMPY.replace(replacement_dict).values
                cmap = copy.copy(mpl.cm.get_cmap(cmap_name))
                cmap.set_bad("white")
                ax.set_axis_off()
                sns.heatmap(V, cmap=cmap, mask=V == np.nan, ax=ax, vmax=1, vmin=0,
                                cbar=None, cbar_ax=None, rasterized=True)

            fig.tight_layout()
            plt.savefig(f'/home/SECONDGRAM/temp/{k}_{i}_{k2}.png', dpi=200)
            plt.close()

            image1 = plt.imread(f'/home/SECONDGRAM/temp/template_background.png')
            image2 = plt.imread(f'/home/SECONDGRAM/temp/{k}_{i}_{k2}.png')
            fig, ax = plt.subplots()
            ax.imshow(image2)
            ax.imshow(image1, alpha=0.15)
            ax.set_axis_off()
            fig.tight_layout()
            plt.savefig(f'/home/SECONDGRAM/plots/{k}_{i}_{k2}.png', dpi=300, bbox_inches='tight')
            plt.close()