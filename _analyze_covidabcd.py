import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pylab as plt

# CORR_FOR_SITES = True

dfk = pd.read_csv('/Users/danilo/f/project_sarah_covidabcd/yabcdcovid19questionnaire01.txt', delimiter='\t')

dfp = pd.read_csv('/Users/danilo/f/project_sarah_covidabcd/pabcdcovid19questionnaire01.txt', delimiter='\t')


for item in dfk.isna().sum(0): print(item) 



# KIDS
# remove all/many NaN columns
my_thresh = int(dfk.shape[0] * 0.5)  # make sure we have at least 10% present values
dfk.dropna(thresh=my_thresh, axis=1, inplace=True)

# real data starting at: dfk.interview_age
dfk_hdr = dfk.iloc[0, :]
dfk = dfk.iloc[1:, :]  # remove header line that is top row (strings !)

dfk.sex = dfk.sex.replace('M', 1).replace('F', 0)

dfk_todrop = ['eventname', 'collection_title', 'study_cohort_name']
dfk_noresponse = [  # anything bigger than 500 is set to NaN
	'ple_close_fam_cv', 
]

dfk.bedtime_routine_cv = pd.to_numeric(dfk.bedtime_routine_cv, errors='coerce')


dfk = dfk.drop(dfk_todrop, axis=1)

idx_nums_start = np.where(dfk.columns == 'sex')[0][0]

for cur_col in dfk.columns[idx_nums_start:]:  # ~3 hours on MBP
  if dfk[cur_col].dtype.name != 'category':
    continue
  print(cur_col)
  print(dfk[cur_col].dtype)
  dumb = pd.get_dummies(dfk[cur_col], prefix=cur_col + '_')
  if dumb.shape[-1] > 50:  # let's not blow up the dimensions too much
    continue
  print(dumb.dtypes)
  print('Expanding cateorical column to dummies: %s' % cur_col)
  dfk = dfk.join(dumb)

for cur_col in dfk.columns[idx_nums_start:]:
	print(cur_col)
	dfk[cur_col] = pd.to_numeric(dfk[cur_col], errors='coerce')
print(dfk.dtypes)

for cur_col in dfk_noresponse:
	dfk[cur_col].replace(999.0, np.nan, inplace=True)

np.random.seed(0)
for i_col in np.arange(idx_nums_start, dfk.shape[-1]):
    print(dfk.columns[i_col])
    vals = np.array(dfk.iloc[:, i_col])
    vals_set = vals[~np.isnan(vals)]
    inds_nan = np.where(np.isnan(vals))[0]
    n_misses = len(inds_nan)
    inds_repl = np.random.randint(0, len(vals_set), n_misses)

    vals[inds_nan] = vals_set[inds_repl]
    assert np.all(np.isfinite(vals))
    dfk.iloc[:, i_col] = vals

assert dfk.isna().sum().sum() == 0
assert np.sum(dfk.iloc[:, idx_nums_start:] > 500).sum() == 0



# PARENTS
# real data starting at: dfk.interview_age
dfp_hdr = dfp.iloc[0, :]
dfp = dfp.iloc[1:, :]  # remove header line that is top row (strings !)

dfk = dfk.sort_values('subjectkey')
dfp = dfp.sort_values('subjectkey')

assert np.all(dfk.subjectkey.values == dfp.subjectkey.values)  # make sure matched !!

dfp.dropna(thresh=my_thresh, axis=1, inplace=True)

dfp.sex = dfp.sex.replace('M', 1).replace('F', 0)

dfp_todrop = ['eventname', 'collection_title', 'study_cohort_name',
	'school_close_date_cv']

dfp = dfp.drop(dfp_todrop, axis=1)

idx_nums_start = np.where(dfp.columns == 'sex')[0][0]

for cur_col in dfp.columns[idx_nums_start:]:  # ~3 hours on MBP
  if dfp[cur_col].dtype.name != 'category':
    continue
  print(cur_col)
  print(dfp[cur_col].dtype)
  dumb = pd.get_dummies(dfp[cur_col], prefix=cur_col + '_')
  if dumb.shape[-1] > 50:  # let's not blow up the dimensions too much
    continue

  print(dumb.dtypes)
  print('Expanding cateorical column to dummies: %s' % cur_col)
  dfp = dfp.join(dumb)

for cur_col in dfp.columns[idx_nums_start:]:
	print(cur_col)
	dfp[cur_col] = pd.to_numeric(dfp[cur_col], errors='coerce')
print(dfp.dtypes)

for cur_col in dfp.columns[idx_nums_start:]:
	dfp[cur_col].replace(999.0, np.nan, inplace=True)
	dfp[cur_col].replace(777.0, np.nan, inplace=True)

np.random.seed(0)
for i_col in np.arange(idx_nums_start, dfp.shape[-1]):
    print(dfp.columns[i_col])
    vals = np.array(dfp.iloc[:, i_col])
    vals_set = vals[~np.isnan(vals)]
    inds_nan = np.where(np.isnan(vals))[0]
    n_misses = len(inds_nan)
    inds_repl = np.random.randint(0, len(vals_set), n_misses)

    vals[inds_nan] = vals_set[inds_repl]
    assert np.all(np.isfinite(vals))
    dfp.iloc[:, i_col] = vals

assert dfp.isna().sum().sum() == 0
assert np.sum(dfp.iloc[:, idx_nums_start:] > 500).sum() == 0



# load baseline data now
import os
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Versions/Current/Resources/"
import rpy2.robjects as robjects 
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

converted_file = '/Users/danilo/f/project_sarah_covidabcd/nda3_baselineevent.csv'


pandas2ri.activate()
readRDS = robjects.r['readRDS']
dfbase = readRDS('/Users/danilo/f/project_sarah_covidabcd/nda3.0_2021jan.Rds')

dfbase.sex = dfk.sex.replace('M', 1).replace('F', 0)

dfbase_ = dfbase[dfbase.eventname=='baseline_year_1_arm_1']
dfbase_.to_csv(converted_file)

# make sure we have the right demographic info for subsetting later
income_below_50k = ((dfbase_.demo_comb_income_p=='Less than $5 000') |
  (dfbase_.demo_comb_income_p=='$5 000 through $11 999') |
  (dfbase_.demo_comb_income_p=='$12 000 through $15 999') |
  (dfbase_.demo_comb_income_p=='$16 000 through $24 999') |
  (dfbase_.demo_comb_income_p=='$25 000 through $34 999') |
  (dfbase_.demo_comb_income_p=='$35 000 through $49 999')).astype(int)
dfbase_['income_below_50k'] = income_below_50k

# build continuous income variable, too
income_cont = np.ones(len(dfbase_.demo_comb_income_p)) * -1
income_cont[dfbase_.demo_comb_income_p == 'Less than $5 000'] = 0
income_cont[dfbase_.demo_comb_income_p == '$5 000 through $11 999'] = 1
income_cont[dfbase_.demo_comb_income_p == '$12 000 through $15 999'] = 2
income_cont[dfbase_.demo_comb_income_p == '$16 000 through $24 999'] = 3
income_cont[dfbase_.demo_comb_income_p == '$25 000 through $34 999'] = 4
income_cont[dfbase_.demo_comb_income_p == '$35 000 through $49 999'] = 5
income_cont[dfbase_.demo_comb_income_p == '$50 000 through $74 999'] = 6
income_cont[dfbase_.demo_comb_income_p == '$75 000 through $99 999'] = 7
income_cont[dfbase_.demo_comb_income_p == '$100 000 through $199 999'] = 8
income_cont[dfbase_.demo_comb_income_p == '$200 000 and greater'] = 9
income_cont[dfbase_.demo_comb_income_p == "Don't know"] = 5  # mean
income_cont[dfbase_.demo_comb_income_p == 'Refuse to answer'] = 5  # mean
income_cont[dfbase_.demo_comb_income_p.isna()] = 5  # mean
income_cont[dfbase_.demo_comb_income_p == '$200 000 and greater'] = 10

assert np.sum(income_cont == -1) == 0
dfbase_['income_cont'] = income_cont


dumb = pd.get_dummies(dfbase_.site_id_l)
dfbase_ = dfbase_.join(dumb)

any_higher_ed = ((dfbase_.demo_prnt_ed_p=='Bachelor\'s degree (ex. BA  AB  BS  BBS)') |
  (dfbase_.demo_prnt_ed_p=='Master\'s degree (ex. MA  MS  MEng  MEd  MBA)') |
  (dfbase_.demo_prnt_ed_p=='Professional School degree (ex. MD  DDS  DVN  JD)') |
  (dfbase_.demo_prnt_ed_p=='Doctoral degree (ex. PhD  EdD)')).astype(int)
dfbase_['demo_prnt_any_higher_ed'] = any_higher_ed

any_higher_ed = ((dfbase_.demo_prtnr_ed_p=='Bachelor\'s degree (ex. BA  AB  BS  BBS)') |
  (dfbase_.demo_prtnr_ed_p=='Master\'s degree (ex. MA  MS  MEng  MEd  MBA)') |
  (dfbase_.demo_prtnr_ed_p=='Professional School degree (ex. MD  DDS  DVN  JD)') |
  (dfbase_.demo_prtnr_ed_p=='Doctoral degree (ex. PhD  EdD)')).astype(int)
dfbase_['demo_prtnr_any_higher_ed'] = any_higher_ed

# pd.get_dummies(dfbase_.neighb_phenx_1r_p)  # neighborhood safety (prnt)
# pd.get_dummies(dfbase_.neighb_phenx)  # neighborhood safety (youth)


# family history of mental health aspects: build summary score
famhx_ss = dfbase_.iloc[:, dfbase_.columns.str.contains('famhx_ss')]
famhx_ss = famhx_ss.replace({
  'no problem endorsed': 0, 'no': 0, 'none': 0})
famhx_ss = famhx_ss.replace({
  'problem endorsed': 1, 'father only': 1, 'mother only': 1,
  'yes': 1, 'both': 1})
for cur_col in famhx_ss.columns:
  print(cur_col)
  famhx_ss[cur_col] = pd.to_numeric(famhx_ss[cur_col], errors='coerce')
famhx_ss = famhx_ss.values
famhx_ss[np.isnan(famhx_ss)] = 0
famhx_ss[famhx_ss > 1] = 1
famhx_ss[famhx_ss < 0] = 0
dfbase_['famhx_ss'] = np.sum(famhx_ss, axis=1)


# remove MRI-related information
dfbase__ = dfbase_.iloc[:, ~dfbase_.columns.str.contains('mri')]

my_thresh = int(dfbase__.shape[0] * 0.5)
dfbase__.dropna(thresh=my_thresh, axis=1, inplace=True)  # down to 6202 columns

for cur_col in dfbase__.columns[3:]:  # ~3 hours on MBP
  if dfbase__[cur_col].dtype.name != 'category':
    continue
  print(cur_col)
  print(dfbase__[cur_col].dtype)
  dumb = pd.get_dummies(dfbase__[cur_col], prefix=cur_col + '_')
  if dumb.shape[-1] > 50:  # let's not blow up the dimensions too much
    continue

  print(dumb.dtypes)
  print('Expanding cateorical column to dummies: %s' % cur_col)
  dfbase__ = dfbase__.join(dumb)


for cur_col in dfbase__.columns[3:]:  # deletes Category columns anyways
  print(cur_col)
  dfbase__[cur_col] = pd.to_numeric(dfbase__[cur_col], errors='coerce')

dfbase__.dropna(thresh=my_thresh, axis=1, inplace=True)  # down to 17429 columns

assert np.all(dfbase__.src_subject_id == dfbase_.src_subject_id)
dfbase__['subjectid'] = dfbase_['subjectid']

dfbase__ = dfbase__.sort_values('subjectid')


dfbase__ = dfbase__.loc[dfbase__.subjectid.isin(dfk.subjectkey)]
dfbase__ = dfbase__.loc[dfbase__.subjectid.isin(dfp.subjectkey)]

for cur_col in dfbase__.columns[3:]:
  dfbase__[cur_col].replace(999.0, np.nan, inplace=True)
  dfbase__[cur_col].replace(777.0, np.nan, inplace=True)

np.random.seed(0)
for i_col in np.arange(3, dfp.shape[-1]):
    print(dfbase__.columns[i_col])
    vals = np.array(dfbase__.iloc[:, i_col])
    vals_set = vals[~np.isnan(vals)]
    inds_nan = np.where(np.isnan(vals))[0]
    n_misses = len(inds_nan)
    print(n_misses)
    inds_repl = np.random.randint(0, len(vals_set), n_misses)

    vals[inds_nan] = vals_set[inds_repl]
    assert np.all(np.isfinite(vals))
    dfbase__.iloc[:, i_col] = vals


dfbase__.drop(['income_below_50k', 'demo_prnt_any_higher_ed',
  'demo_prtnr_any_higher_ed'], inplace=True, axis=1)



# match the Covid dataset from child and parents

dfk_1tp = dfk.drop_duplicates(subset='subjectkey')
dfk_1tp = dfk_1tp[dfk_1tp.subjectkey.isin(dfbase__.subjectid)]
assert np.all(dfbase__.subjectid.values==dfk_1tp.subjectkey.values)
dfp_1tp = dfp.drop_duplicates(subset='subjectkey')
dfp_1tp = dfp_1tp[dfp_1tp.subjectkey.isin(dfbase__.subjectid)]
assert np.all(dfbase__.subjectid.values==dfp_1tp.subjectkey.values)

# takeouts to facilitate interpretation
dfk_1tp.drop('interview_age', axis=1, inplace=True)
dfp_1tp.drop('interview_age', axis=1, inplace=True)
dfk_1tp.drop('sex', axis=1, inplace=True)
dfp_1tp.drop('sex', axis=1, inplace=True)

Y = np.hstack([dfk_1tp.iloc[:, 6:].values, dfp_1tp.iloc[:, 6:].values])
Y_cols = list(dfk_1tp.iloc[:, 6:].columns) + list(dfp_1tp.iloc[:, 6:].columns)
X = dfbase__.iloc[:, 3:].values
X_cols = list(dfbase__.iloc[:, 3:].columns)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.nan_to_num(X)
Y = np.nan_to_num(Y)

# dump just in case
import joblib
joblib.dump(dfbase__, 'dump_dfbase__')
joblib.dump(X, 'dump_X')
joblib.dump(Y, 'dump_Y')
joblib.dump(Y_cols, 'dump_Y_cols')
joblib.dump(X_cols, 'dump_X_cols')

dfbase__ = joblib.load('dump_dfbase__')
X = joblib.load('dump_X')
Y = joblib.load('dump_Y')
Y_cols = joblib.load('dump_Y_cols')
X_cols = joblib.load('dump_X_cols')


# make PCA step well-behaved (p < n)
cnt_mainitem = pd.DataFrame(X).apply(
  lambda x: np.array(x.value_counts())[0] , axis=0)
idx_changes = np.where(cnt_mainitem < (len(X) - 250))[0]  # diff in <250 people?
X = X[:, idx_changes]
X_cols = np.array(X_cols)[idx_changes]

X_ss = StandardScaler().fit_transform(X) 
Y_ss = StandardScaler().fit_transform(Y)


if 'CORR_FOR_SITES' in locals():
  from nilearn.signal import clean

  Y_ss = clean(Y_ss, detrend=False, confounds=pd.get_dummies(dfbase__.site_location))


x_pca = PCA(n_components=250)
y_pca = PCA(n_components=50)

X_ss_pca = x_pca.fit_transform(X_ss)
Y_ss_pca = y_pca.fit_transform(Y_ss)

STOP



n_permutations = 1000
cur_X = np.array(X_ss_pca)
cur_Y = np.array(Y_ss_pca)
perm_rs = np.random.RandomState(0)
perm_Rs = []
perm_scores = []
n_except = 0
for i_iter in range(n_permutations):
    print(i_iter + 1)

    cur_X_perm = np.array([perm_rs.permutation(sub_entry) for sub_entry in cur_X])

    # same procedure, only with permuted subjects on the right side
    try:
        perm_cca = CCA(n_components=n_comp, scale=True)

        perm_cca.fit(cur_X_perm, cur_Y)

        perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
            zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
        cur_score = perm_cca.score(cur_X, cur_Y)
        print(np.sort(perm_R)[::-1][:10])
        print(cur_score)
        perm_Rs.append(perm_R)
        perm_scores.append(cur_score)
    except:
        n_except += 1
        perm_Rs.append(np.zeros(n_keep))
perm_Rs = np.array(perm_Rs)


pvals = []
for i_coef in range(n_comp):
    cur_pval = (1. + np.sum(perm_Rs[1:, 0] > actual_Rs[i_coef])) / n_permutations
    pvals.append(cur_pval)
    print(cur_pval)
pvals = np.array(pvals)
print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))

# 9 CCs are significant at p<0.05
# 9 CCs are significant at p<0.01
# 0 CCs are significant at p<0.001





# CCA
from sklearn.cross_decomposition import CCA, PLSRegression

n_comp = 10
# est = PLSRegression(n_components=n_comp, scale=False)  # ALWAYS de-means internally !!
est = CCA(n_components=n_comp, scale=True)  # ALWAYS de-means internally !!
est.fit(X_ss_pca, Y_ss_pca)
# est.fit(X_ss, Y_ss)

actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(est.x_scores_.T, est.y_scores_.T)])
inds_max_to_min = np.argsort(actual_Rs)[::-1]
actual_Rs_sorted = actual_Rs[inds_max_to_min]
print(actual_Rs_sorted[:n_comp])


OUTDIR = 'results_crossdecomp_repl3'

top_k = 25  # per component

for i_c in range(n_comp):

  x_comp_weights = x_pca.inverse_transform(est.x_loadings_.T)[i_c]
  idx_max = np.argsort(np.abs(x_comp_weights))[::-1]
  idx_max_top = idx_max[:top_k]

  df_left = pd.DataFrame(np.atleast_2d(x_comp_weights[idx_max]).T,
    index=np.array(X_cols)[idx_max],
      columns=[i_c + 1])
  df_left_top = pd.DataFrame(np.atleast_2d(x_comp_weights[idx_max_top]).T,
    index=np.array(X_cols)[idx_max_top],
      columns=[i_c + 1])

  plt.figure(figsize=(12, 10))
  ax = sns.heatmap(df_left_top.T, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.25}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  # ax.set_xticks(np.arange(0, len(idx_max_top)) + 0.5)  
  # ax.set_xticklabels(df_plot_k.columns, fontsize=7)
  # b, t = plt.ylim() # discover the values for bottom and top
  # b += 0.5 # Add 0.5 to the bottom
  # t -= 0.5 # Subtract 0.5 from the top
  # plt.ylim(b, t) # update the ylim(bottom, top) values


  plt.ylabel('mode number')
  plt.tight_layout()
  plt.title('Baseline-Covid Projection: top %i from left component %i' % (top_k, i_c + 1))

  plt.savefig('%s/cca_basecov_top%i_c%i_left.png' % (OUTDIR, top_k, i_c + 1), DPI=600)
  # plt.savefig('results_crossdecomp/cca_basecov_top%i_c%i_left.pdf' % (top_k, i_c + 1))
  df_left.to_csv('%s/cca_basecov_topall_c%i_left.csv' % (OUTDIR, i_c + 1))



  y_comp_weights = y_pca.inverse_transform(est.y_loadings_.T)[i_c]
  idx_max = np.argsort(np.abs(y_comp_weights))[::-1]
  idx_max_top = idx_max[:top_k]

  df_right = pd.DataFrame(np.atleast_2d(y_comp_weights[idx_max]).T,
    index=np.array(Y_cols)[idx_max],
      columns=[i_c + 1])
  df_right_top = pd.DataFrame(np.atleast_2d(y_comp_weights[idx_max_top]).T,
    index=np.array(Y_cols)[idx_max_top],
      columns=[i_c + 1])

  plt.figure(figsize=(12, 10))
  ax = sns.heatmap(df_right_top.T, cbar=True, linewidths=.75,
                 cbar_kws={'shrink': 0.25}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
                 square=True,
                 cmap=plt.cm.RdBu_r, center=0)

  ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
  ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
  # ax.set_xticks(np.arange(0, len(dfp.columns) - 8) + 0.5)  
  # ax.set_xticklabels(dfp.columns[8:], fontsize=6)
  # b, t = plt.ylim() # discover the values for bottom and top
  # b += 0.5 # Add 0.5 to the bottom
  # t -= 0.5 # Subtract 0.5 from the top
  # plt.ylim(b, t) # update the ylim(bottom, top) values


  plt.ylabel('mode number')
  plt.tight_layout()
  plt.title('Baseline-Covid Projection: top %i from right component %i' % (top_k, i_c + 1))

  plt.savefig('%s/cca_basecov_top%i_c%i_right.png' % (OUTDIR, top_k, i_c + 1), DPI=600)
  # plt.savefig('results_crossdecomp/cca_basecov_top%i_c%i_right.pdf' % (top_k, i_c + 1))
  df_right.to_csv('%s/cca_basecov_topall_c%i_right.csv' % (OUTDIR, i_c + 1))


plt.close('all')


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  hb = ax.hexbin(est.x_scores_[:, i_mode], est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='viridis')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'baseline canonical variates', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-5, +5); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')

# number mental health issues
# tar_y = StandardScaler().fit_transform(y_famhx_ss[:, None])[:, 0]
tar_y = y_famhx_ss
max_mode = 9
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  hb = ax.hexbin(tar_y, est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='Reds')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'number of mental health issues', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-5, +5); ax.set_xlim(0, 30)
  plt.show()
  plt.savefig('%s/cca_hexbin_mentalhealth_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_mentalhealth_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')

# with income
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  rs = np.random.RandomState(i_mode)
  jitter = rs.randn(len(dfbase__))
  hb = ax.hexbin(dfbase__.income_cont + jitter,
                 est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'low vs. high family income', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-3, +3); ax.set_xlim(0, 11)
  plt.show()
  plt.savefig('%s/cca_hexbin_income_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_income_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')



max_mode = 9
tar_y = StandardScaler().fit_transform(dfbase__.perc_at_least_one_dose[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  rs = np.random.RandomState(i_mode)
  jitter = rs.randn(len(dfbase__))
  hb = ax.hexbin(tar_y + jitter,
                 est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'percentage of people with at least 1 dose', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-3, +3); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_least1dose_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_least1dose_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')


max_mode = 9
tar_y = StandardScaler().fit_transform(dfbase__.cases_per100k[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  rs = np.random.RandomState(i_mode)
  jitter = rs.randn(len(dfbase__))
  hb = ax.hexbin(tar_y + jitter,
                 est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'#cases', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-3, +3); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_cases_per100k_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_cases_per100k_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')


max_mode = 9
tar_y = StandardScaler().fit_transform(dfbase__.deaths_per100k[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  rs = np.random.RandomState(i_mode)
  jitter = rs.randn(len(dfbase__))
  hb = ax.hexbin(tar_y + jitter,
                 est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'#deaths', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-3, +3); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_deaths_per100k_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_deaths_per100k_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
tar_y = StandardScaler().fit_transform(
  dfbase__.reshist_addr1_leadrisk_pov[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  hb = ax.hexbin(tar_y, est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'percentage of people below poverty level', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-5, +5); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_reshist_addr1_leadrisk_pov_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_reshist_addr1_leadrisk_pov_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
tar_y = StandardScaler().fit_transform(
  dfbase__.reshist_addr1_adi_b138[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  hb = ax.hexbin(tar_y, est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'percentage of people below poverty level', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-5, +5); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_reshist_addr1_adi_b138_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_reshist_addr1_adi_b138_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
tar_y = StandardScaler().fit_transform(
  dfbase__.reshist_addr1_adi_pov[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  hb = ax.hexbin(tar_y, est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'percentage of people below poverty level', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-5, +5); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_reshist_addr1_adi_pov_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_reshist_addr1_adi_pov_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')




import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
tar_y = dfbase__.screentime_ss_weekend
# tar_y = StandardScaler().fit_transform(
#   dfbase__.screentime_ss_weekend[:, None])[:, 0]
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 250.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  hb = ax.hexbin(tar_y, est.y_scores_[:, i_mode],
                 gridsize=50,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'weekend screen time (child)', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  # ax.set_ylim(-5, +5); ax.set_xlim(-5, +5)
  plt.show()
  plt.savefig('%s/cca_hexbin_screentime_ss_weekend_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_screentime_ss_weekend_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
  plt.savefig('%s/cca_hexbin_screentime_ss_weekend_top%i_mode%i.eps' % (OUTDIR, max_mode, i_mode + 1))
  plt.savefig('%s/cca_hexbin_screentime_ss_weekend_top%i_mode%i.tiff' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

max_mode = 9
# tar_y = StandardScaler().fit_transform(
#   dfbase__.demo_roster_p[:, None])[:, 0]
tar_y = dfbase__.demo_roster_p
for i_mode in range(max_mode):
  n_simus = len(est.x_scores_) / 125.
  fig, ax = plt.subplots(1, 1, figsize=(6, 6))

  rs = np.random.RandomState(i_mode)
  jitter = rs.randn(len(dfbase__))
  hb = ax.hexbin(tar_y + jitter, est.y_scores_[:, i_mode],
                 gridsize=100,
                 norm=plt.matplotlib.colors.Normalize(0, n_simus),
                 mincnt=1,
                 cmap='inferno')

  ax.grid(True)
  # ax.axvline(-np.log10(0.05), color='red', linestyle='--')

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='3%', pad=0.05)

  cb = fig.colorbar(hb, cax=cax, orientation='vertical')
  cb.set_alpha(1)
  cb.draw_all()

  plt.suptitle(f'Population mode {i_mode + 1}: explain variance is %.2f (r2)' % actual_Rs[i_mode],
    fontsize=14, fontweight=100)

  ax.set_xlabel(r'family size', fontsize=12,
                fontweight=150)
  ax.set_ylabel(r'COVID19 canonical variates', fontsize=12,
                fontweight=150)
  ax.set_ylim(-5, +5); ax.set_xlim(0, 12)
  plt.show()
  plt.savefig('%s/cca_hexbin_demo_roster_p_top%i_mode%i.png' % (OUTDIR, max_mode, i_mode + 1), DPI=600)
  plt.savefig('%s/cca_hexbin_demo_roster_p_top%i_mode%i.pdf' % (OUTDIR, max_mode, i_mode + 1))
  plt.savefig('%s/cca_hexbin_demo_roster_p_top%i_mode%i.eps' % (OUTDIR, max_mode, i_mode + 1))
  plt.savefig('%s/cca_hexbin_demo_roster_p_top%i_mode%i.tiff' % (OUTDIR, max_mode, i_mode + 1))
plt.close('all')


# Site-specific analyses
sinfo = pd.read_csv(
  '/Users/danilo/f/project_sarah_covidabcd/siteid_location.csv') 
s2info = pd.read_excel(
  '/Users/danilo/f/project_sarah_covidabcd/siteid_location_2.xlsx')
s3info = pd.read_excel(
  '/Users/danilo/f/project_sarah_covidabcd/siteid_location_3.xlsx')
s3info = s3info.head(21)

# sinfo.s2in
dfbase__['election_2020'] = np.nan
dfbase__['site_name'] = np.nan
dfbase__['site_location'] = np.nan
dfbase__['cases_per100k'] = np.nan
dfbase__['deaths_per100k'] = np.nan
dfbase__['perc_at_least_one_dose'] = np.nan
for sname in sinfo.site_number[:-1]:
  print(sname)
  dfbase__.loc[dfbase__[sname]==1, 'perc_at_least_one_dose'] = s3info[
    s3info.site_number==sname].perc_at_least_one_dose.values[0]

  dfbase__.loc[dfbase__[sname]==1, 'election_2020'] = sinfo[
    sinfo.site_number==sname].election_2020.values[0]

  dfbase__.loc[dfbase__[sname]==1, 'site_name'] = sinfo[
    sinfo.site_number==sname].description.values[0]

  dfbase__.loc[dfbase__[sname]==1, 'site_location'] = sinfo[
    sinfo.site_number==sname].site_location.values[0]

  dfbase__.loc[dfbase__[sname]==1, 'cases_per100k'] = s2info[
    sinfo.site_number==sname].cases_per100k.values[0]

  dfbase__.loc[dfbase__[sname]==1, 'deaths_per100k'] = s2info[
    sinfo.site_number==sname].deaths_per100k.values[0]

assert dfbase__['election_2020'].isna().sum() == 0
assert dfbase__['site_name'].isna().sum() == 0
assert dfbase__['site_location'].isna().sum() == 0

print(dfbase__['election_2020'].value_counts())


# boxplot per site
df_out_y = pd.DataFrame()
for i_cc in range(n_comp):

  for i_site, str_site in enumerate(str_site_names):
    idx_site = np.where(dfbase__.site_name == str_site)[0]

    df_site = pd.DataFrame(est.y_scores_[idx_site, i_cc], columns=['variate'])
    df_site['site'] = i_site 
    df_site['mode'] = i_cc

    df_out_y = df_out_y.append(df_site)

plt.figure(figsize=(14, 4))
sns.boxplot(y='variate', x='site', data=df_out_y, hue='site')

df_out_y = pd.DataFrame()
for i_cc in range(n_comp):

  for i_site, str_site in enumerate(str_site_names):
    idx_site = np.where(dfbase__.site_name == str_site)[0]

    df_site = pd.DataFrame(est.y_scores_[idx_site, i_cc], columns=['variate'])
    df_site['site'] = i_site 
    df_site['mode'] = i_cc

    df_out_y = df_out_y.append(df_site)

plt.figure(figsize=(18, 4))
ax = sns.boxplot(y='variate', x='mode', data=df_out_y, hue='mode')



for i_site, str_site in enumerate(str_site_names):
  idx_site = np.where(dfbase__.site_name == str_site)[0]

  site_X = est.y_scores_[idx_site, :]
  plt.figure(figsize=(18, 4))
  plt.boxplot(site_X) 
  plt.xlabel('mode')
  plt.ylabel('variate')
  plt.ylim(-2.5, +2.5)

  plt.savefig(OUTDIR + f'/boxplot_site{i_site + 1}_{str_site}.png') 

plt.close('all')


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(fit_intercept=True, multi_class='ovr')
y = dfbase__.site_name
lr.fit(est.y_scores_, y)
plt.figure()
plt.imshow(lr.coef_, cmap=plt.cm.RdBu_r)
plt.colorbar()
plt.yticks(np.arange(len(lr.classes_)), lr.classes_)  
plt.xticks(np.arange(10), np.arange(10) + 1)
plt.xlabel('mode')
plt.title('Which modes are most specific for which site?\n(logistic regression analysis to distinguish 22 sites)')
plt.savefig(OUTDIR + f'/headmap_classif_22sites.png') 





elec_names = dfbase__['election_2020'].unique()  # not too beautiful
for i_mode in range(n_comp):

  site_X = []
  site_cols = []
  for i_elec, str_elec in enumerate(elec_names):
    idx_elec = np.where(dfbase__['election_2020'] == str_elec)[0]
    site_X.append(est.y_scores_[idx_elec, i_mode])
  plt.figure(figsize=(18, 8))
  plt.boxplot(site_X) 
  # plt.xlabel('mode')
  plt.ylabel(f'variate of mode {i_mode + 1}')
  plt.ylim(-5.0, +5.0)
  plt.xticks(np.arange(len(elec_names)) + 1, elec_names, rotation=45)
  plt.tight_layout()

  plt.savefig(OUTDIR + f'/boxplot_election_2020_mode{i_mode + 1}.png') 
  plt.savefig(OUTDIR + f'/boxplot_election_2020_mode{i_mode + 1}.pdf') 
  plt.savefig(OUTDIR + f'/boxplot_election_2020_mode{i_mode + 1}.eps') 
plt.close('all')


y_dem = np.array(dfbase__['election_2020'] == 'democrat') 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(fit_intercept=True)
bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(y_dem), len(y_dem))

  lr.fit(est.y_scores_[bs_idx], y_dem[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], y_dem[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(y_dem)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], y_dem[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('republican vs. democrat\n(mean effect +/- bootstrap population intervals)')
plt.title('Classifying political affiliation from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif.png') 
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif.pdf') 
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif.eps') 
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif.tiff')

plt.ylim(-1, +1)
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif_fixedylim.png') 
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif_fixedylim.pdf') 
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif_fixedylim.eps') 
plt.savefig(OUTDIR + f'/boxplot_election_2020_classif_fixedylim.tiff')
plt.close('all')




# this item influence the original CCA analysis
y_rac_disc = np.squeeze(Y[:, 'witness_rac_disc_cv' == Y_cols])
y_rac_disc = np.array(y_rac_disc >= 3, dtype=np.int)  # 0 = Never ; 1 = Rarely ; 2 = Occasionally ; 3 = Frequently ; 4 = Very frequently
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(fit_intercept=True)
bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(y_rac_disc), len(y_rac_disc))

  lr.fit(est.y_scores_[bs_idx], y_rac_disc[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], y_rac_disc[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(y_rac_disc)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], y_rac_disc[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('never/occasionally vs. frequently\n(mean effect +/- bootstrap population intervals)')
plt.title('Classifying racial discrimination from COVID19 items')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_witness_rac_disc_classif.png') 
plt.close('all')




y_worry = np.squeeze(
  StandardScaler().fit_transform(Y[:, np.array(Y_cols) == 'worry_y_cv']))

can_variate_base_1 = est.x_scores_.T[0, :]

meta_level = dfbase__.site_name.replace(
  dfbase__.site_name.unique(),
  np.arange(len(dfbase__.site_name.unique())))
n_sites = 22

y_outcome = y_worry

with pm.Model() as hierarchical_model1:
  BURN_IN_STEPS = 500
  MCMC_STEPS = 1000

  # linear model
  interc = pm.Normal('interc', mu=0, sd=1, shape=1)

  beta1 = pm.Normal('CCA1', mu=0, sd=1, shape=n_sites)

  beh_est = interc + beta1[meta_level.values] * can_variate_base_1[:, None]

  eps = pm.HalfCauchy('eps', 5)  # Model error
  group_like = pm.Normal('beh_like', mu=beh_est, sd=eps, observed=y_outcome)


with hierarchical_model1:
    hierarchical_trace1 = pm.sample(draws=MCMC_STEPS, n_init=BURN_IN_STEPS,
        # chains=1, cores=1, progressbar=True, random_seed=3)
        chains=1, cores=1, progressbar=True, random_seed=[4])
hierarchical_trace1 = hierarchical_trace1
t = hierarchical_trace1


# date base prediction
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import ShuffleSplit, cross_val_score   

assert len(dt.dt.year.unique()) == 1  # everybody from same year ?
dt = pd.to_datetime(dfp_1tp.interview_date, infer_datetime_format=True)
doy_p = dt.dt.dayofyear
dt = pd.to_datetime(dfk_1tp.interview_date, infer_datetime_format=True)
doy_k = dt.dt.dayofyear


ABC
sstrans = StandardScaler()  # use same for kids and parent dates, but fit on kids
# sstrans_k = StandardScaler()

doy_k_ss = sstrans.fit_transform(doy_k.values.reshape(-1, 1))
doy_p_ss = sstrans.transform(doy_p.values.reshape(-1, 1))

lr = LinearRegression(fit_intercept=True)
folder = ShuffleSplit(n_splits=10, test_size=0.1)
acc = cross_val_score(
  estimator=lr,
  X=Y_ss, y=doy_k_ss,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=Y_ss, y=doy_k_ss)
print(lr.score(X=Y_ss, y=doy_k_ss))
# 0.2702992923720472

for kind in [0, 1]:
  tar_y = doy_p_ss if kind==0 else doy_k_ss
  SUBJ = 'parent' if kind==0 else 'child'

  bs_coefs = []
  np.random.seed(0)
  for i_bs in range(100):
    bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

    lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
    is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
    print(is_acc)

    print(idx_OOB.sum())
    idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
    oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

    bs_coefs.append(lr.coef_)

  bs_coefs = np.squeeze(bs_coefs)
  plt.figure(figsize=(6, 4))
  plt.boxplot(bs_coefs) 
  plt.xlabel('mode')
  plt.ylabel('earlier vs. later in 2020\n(mean effect +/- bootstrap population intervals)')
  plt.title(f'Relation of COVID19 items to interview time during pandemic\n{SUBJ}')
  # plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
  plt.tight_layout()
  plt.savefig(OUTDIR + f'/boxplot_interviewtime_{SUBJ}_regr.png') 
  plt.close('all')




# classify plitical orientation of the state
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

y_repub = dfbase__['election_2020'] == 'republican'
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=Y_ss, y=y_repub,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=Y_ss, y=y_repub)


idx_max = np.argsort(np.abs(lr.coef_[0] ))[::-1]
idx_max_top = idx_max[:top_k]

df_right = pd.DataFrame(np.atleast_2d(lr.coef_[0][idx_max]),
  columns=np.array(Y_cols)[idx_max],
    index=[i_c + 1])
df_right_top = pd.DataFrame(np.atleast_2d(lr.coef_[0][idx_max_top]),
  columns=np.array(Y_cols)[idx_max_top],
    index=[i_c + 1])

plt.figure(figsize=(12, 10))
ax = sns.heatmap(df_right_top, cbar=True, linewidths=.75,
               cbar_kws={'shrink': 0.25}, #'orientation': 'horizontal'},  #, 'label': 'Functional coupling deviation'},
               square=True,
               cmap=plt.cm.RdBu_r, center=0)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
# ax.set_xticks(np.arange(0, len(dfp.columns) - 8) + 0.5)  
# ax.set_xticklabels(dfp.columns[8:], fontsize=6)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values


plt.ylabel('mode number')
plt.tight_layout()

plt.savefig('results_crossdecomp/classif_republicans_Yss.png', DPI=600)
# plt.savefig('results_crossdecomp/cca_basecov_top%i_c%i_right.pdf' % (top_k, i_c + 1))
df_right.T.to_csv('results_crossdecomp/classif_republicans_Yss.csv')





# If you lost all your current source(s) of household income (your paycheck public assistance or other forms of income) how long could you continue to live at your current address and standard of living? 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

y_atleast1month = np.squeeze(Y[:, Y_cols == 'fam_exp7_v2_cv'] != 1.0).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=Y_ss, y=y_atleast1month,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=Y_ss, y=y_atleast1month)
print(lr.score(X=Y_ss, y=y_atleast1month))
# 1.0

tar_y = y_atleast1month
bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('family sustains less than 4 weeks vs. at least 1 month\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Relation of COVID19 items to living standard without new income\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_living_standard_classif.png') 
plt.close('all')

for i_mode in range(2):
  df_coefs = pd.DataFrame()
  # df_coefs['c'] = bs_coefs
  # df_coefs['m'] = np.squeeze([[f'mode{i + 1}']*100 for i in range(10)]).reshape(-1, 1)
  df_coefs['c'] = bs_coefs[:, i_mode]
  df_coefs['m'] = np.squeeze([f'mode{i_mode + 1}']*100)
  f, ax = plt.subplots(figsize=(6, 12), sharex=True)
  # f, ax = plt.subplots()
  # ax.vlines(x=0, ymin=-1, ymax=40, linestyles='dashed', colors='#999999', 
  #             linewidth=0.7, alpha=0.8)     
  pt.RainCloud(y='c', #y=f'mode{i_mode + 1}', 
              data=df_coefs, orient='v', bw=0.2,
              # hue=dhue,
              # alpha=0.75,
              dodge=True,
              # # palette=social_colors,
              point_size=5.0, linewidth=0.2,
              width_viol=0.35, width_box=0.05,
              )
  ax.set_xticklabels(['living standard without new income'])
  plt.tight_layout()
  ax.set_ylim(-1, +1)
  plt.savefig(OUTDIR + f'/boxplot_living_standard_classif_rain_mode{i_mode + 1}.png') 



# Practiced social distancing in last week ?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'fam_actions_cv___1']).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
# 0.8268023196087111
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Family practiced social distancing during the last week\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_physdist_classif.png') 
plt.close('all')

print(oob_acc)




# I am worried that our family will experience racism or discrimination in relation to coronavirus 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'fam_exp_racism_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Worry about experiencing racism or discrimination?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_worryracism.png') 
plt.savefig(OUTDIR + f'/boxplot_worryracism.pdf') 
plt.savefig(OUTDIR + f'/boxplot_worryracism.eps') 
plt.savefig(OUTDIR + f'/boxplot_worryracism.tiff') 

plt.ylim(-1, +1)
plt.savefig(OUTDIR + f'/boxplot_worryracism_fixedylim.png') 
plt.savefig(OUTDIR + f'/boxplot_worryracism_fixedylim.pdf') 
plt.savefig(OUTDIR + f'/boxplot_worryracism_fixedylim.eps') 
plt.savefig(OUTDIR + f'/boxplot_worryracism_fixedylim.tiff') 
plt.close('all')

print(oob_acc)






# Didn't pay the full amount of the rent or mortgage because you could not afford it? 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'fam_exp4_cv'] == 2).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Struggle paying rent or morgage?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_worryrent.png') 
plt.savefig(OUTDIR + f'/boxplot_worryrent.pdf') 
plt.savefig(OUTDIR + f'/boxplot_worryrent.eps') 
plt.savefig(OUTDIR + f'/boxplot_worryrent.tiff') 

plt.ylim(-1, +1)
plt.savefig(OUTDIR + f'/boxplot_worryrent_fixedylim.png') 
plt.savefig(OUTDIR + f'/boxplot_worryrent_fixedylim.pdf') 
plt.savefig(OUTDIR + f'/boxplot_worryrent_fixedylim.eps') 
plt.savefig(OUTDIR + f'/boxplot_worryrent_fixedylim.tiff') 
plt.close('all')

print(oob_acc)






# family size
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'fam_size_cv']).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('small family vs. large family\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Family size')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_familysize.png') 
plt.close('all')

print(oob_acc)





# Since January 2020 has anyone in your household lost wages sales or work due to the impact of coronavirus on employment business or the economy? 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'fam_wage_loss_cv'] == 2).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  # print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Salary or work loss due to COVID-19?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_workloss.png') 
plt.savefig(OUTDIR + f'/boxplot_workloss.pdf') 
plt.savefig(OUTDIR + f'/boxplot_workloss.eps') 
plt.savefig(OUTDIR + f'/boxplot_workloss.tiff')

plt.ylim(-1, +1)
plt.savefig(OUTDIR + f'/boxplot_workloss_fixedylim.png') 
plt.savefig(OUTDIR + f'/boxplot_workloss_fixedylim.pdf') 
plt.savefig(OUTDIR + f'/boxplot_workloss_fixedylim.eps') 
plt.savefig(OUTDIR + f'/boxplot_workloss_fixedylim.tiff')


print(oob_acc)

for i_mode in range(2):
  df_coefs = pd.DataFrame()
  # df_coefs['c'] = bs_coefs
  # df_coefs['m'] = np.squeeze([[f'mode{i + 1}']*100 for i in range(10)]).reshape(-1, 1)
  df_coefs['c'] = bs_coefs[:, i_mode]
  df_coefs['m'] = np.squeeze([f'mode{i_mode + 1}']*100)
  f, ax = plt.subplots(figsize=(6, 12), sharex=True)
  # f, ax = plt.subplots()
  # ax.vlines(x=0, ymin=-1, ymax=40, linestyles='dashed', colors='#999999', 
  #             linewidth=0.7, alpha=0.8)     
  pt.RainCloud(y='c', #y=f'mode{i_mode + 1}', 
              data=df_coefs, orient='v', bw=0.2,
              # hue=dhue,
              # alpha=0.75,
              dodge=True,
              # # palette=social_colors,
              point_size=5.0, linewidth=0.2,
              width_viol=0.35, width_box=0.05,
              )
  ax.set_xticklabels(['Salary or work loss due to COVID-19?'])
  plt.tight_layout()
  plt.savefig(OUTDIR + f'/boxplot_workloss_rain_mode{i_mode + 1}.png') 





# Increased conflict in family due to COVID-19?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'increased_conflict_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Increased conflict in family due to COVID-19?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_famconflictworry_classif.png') 
plt.close('all')

print(oob_acc)




# Social distanced activity in neighborhood in past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'p_cope_cv___2'] == 1).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Social distanced activity in neighborhood in past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_distactivityneigh_classif.png') 
plt.close('all')

print(oob_acc)






# Connected with others online or by phone ?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'p_cope_cv___8'] == 1).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Connected with others online or by phone?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_connectedothers_classif.png') 
plt.close('all')

print(oob_acc)







# Another adult helps with caregiving?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'caregiver_cv'] == 1).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Another adult helps with caregiving?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_helpwithrearing_classif.png') 
plt.close('all')

print(oob_acc)




# child sleep
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'child_sleep_amount_cv']).astype(np.int)
tar_y = 7 - tar_y  # invert scale: higher number is more hours
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer hours vs. more hours\n(mean effect +/- bootstrap population intervals)')
plt.title(f'How many hours of sleep does the child usually get?')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_sleep_regr.png') 
plt.close('all')

print(oob_acc)




# Parent worried about COVID-19 illness?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'p_worry_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Parent worried about COVID-19 illness?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif.png') 
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif.pdf') 
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif.eps') 
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif.tiff') 

plt.ylim(-1, +1)
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif_fixedylim.png') 
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif_fixedylim.pdf') 
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif_fixedylim.eps') 
plt.savefig(OUTDIR + f'/boxplot_parentworriescovid_classif_fixedylim.tiff')
plt.close('all')



# Child worried about COVID-19 illness?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'child_worry_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child worried about COVID-19 illness?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif.png') 
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif.pdf') 
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif.eps') 
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif.tiff') 

plt.ylim(-1, +1)
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif_fixedylim.png') 
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif_fixedylim.pdf') 
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif_fixedylim.eps') 
plt.savefig(OUTDIR + f'/boxplot_childworriescovid_classif_fixedylim.tiff') 
plt.close('all')





# Avoiding to visit friends or family?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'talk_isolate_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('never/occasionally vs. frequently\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Avoiding to visit friends or family?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_avoidvisits_classif.png') 
plt.close('all')





# Racism or discrimination in relation to coronavirus?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'talk_race_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('never/occasionally vs. frequently\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Racism or discrimination in relation to coronavirus?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_racism_discr_classif.png') 
plt.close('all')






# Child care interfers with work responsibilities?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'work_ability_cv'] >= 3).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('never/some vs. a lot\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child care interfers with work responsibilities?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_work_classif.png') 
plt.close('all')



social_dist_measures = {
'fam_actions_cv___1': 'Engaged in social distancing',
'fam_actions_cv___2': 'Avoided gatherings of 250 people or more',
'fam_actions_cv___3': 'Avoided gatherings of 10 people or more',
'fam_actions_cv___4': 'Avoided visiting family and friends outside our own immediate family',
'fam_actions_cv___5': 'Avoided having people in our home except for a immediate family',
'fam_actions_cv___6': 'Avoided restaurant dining',
'fam_actions_cv___7': 'Avoided restaurant take-out',
'fam_actions_cv___8': 'Avoided grocery store or pharmacies',
'fam_actions_cv___9': 'Avoided stores (not including grocery stores or a pharmacies',
'fam_actions_cv___10': 'Avoided routine doctor visits',
'fam_actions_cv___11': 'Avoided places like gyms malls movie theatres',
'fam_actions_cv___12': 'Avoided taking public transportation',
'fam_actions_cv___13': 'Avoided parks or playgrounds',
'fam_actions_cv___14': 'Wore a mask'}

social_dist_measures = {
'fam_actions_cv___1': 'Engaged in social distancing'}

# social_dist_measures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

for dist_meas, my_title in social_dist_measures.items():

  lr = LogisticRegression(fit_intercept=True) 

  tar_y = np.squeeze(Y[:, Y_cols == dist_meas]).astype(np.int)
  folder = StratifiedKFold(shuffle=True, n_splits=10)
  acc = cross_val_score(
    estimator=lr,
    X=est.y_scores_, y=tar_y,
    cv=folder,
    n_jobs=4)

  print(acc.mean())
  print(acc.std())

  lr.fit(X=est.y_scores_, y=tar_y)
  print(lr.score(X=est.y_scores_, y=tar_y))
  # 1.0

  bs_coefs = []
  np.random.seed(0)
  for i_bs in range(100):
    bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

    lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
    is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
    print(is_acc)

    print(idx_OOB.sum())
    idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
    oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

    bs_coefs.append(lr.coef_)

  bs_coefs = np.squeeze(bs_coefs)
  plt.figure(figsize=(6, 4))
  plt.boxplot(bs_coefs) 
  plt.xlabel('mode')
  plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
  plt.title(my_title)
  # plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
  plt.tight_layout()
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif.png') 
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif.pdf') 
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif.tiff') 
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif.eps') 

  plt.ylim(-1, +1)
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif_fixedylim.png') 
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif_fixedylim.pdf') 
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif_fixedylim.tiff') 
  plt.savefig(OUTDIR + f'/boxplot_socialdist_{dist_meas}_classif_fixedylim.eps') 
  plt.close('all')






# Number of family members diagnosed with COVID-19
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'fam_diag_cv']).astype(np.int)
tar_y = 7 - tar_y  # invert scale: higher number is more hours
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer hours vs. more hours\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Number of family members diagnosed with COVID-19')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_fam_diags_regr.png') 
plt.close('all')




# Ever worried to run out of food?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'work_ability_cv'] == 2).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Ever worried to run out of food?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_food_worry_classif.png') 
plt.close('all')



# Increased expore due to essential job/healthcare job/public transit?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'fam_expose_cv'] == 2).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Increased expore due to essential job/healthcare job/public transit?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_essential_worker_classif.png') 
plt.close('all')






# Number of drink days during last month?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'su_alcohol_cv']).astype(np.int)
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer drink days vs. more drink days\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Number of drink days during last month?')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_alcohol_regr.png') 
plt.close('all')







# Number of drink days during last month?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = np.squeeze(Y[:, Y_cols == 'su_p_cig_use_cv']).astype(np.int)
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer drink days vs. more drink days\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Number of drink days during last month?')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_alcohol_regr.png') 
plt.close('all')







# Child felt alone during past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'felt_alone_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child felt alone during past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_alone_classif.png') 
plt.close('all')





# Child felt always sad during past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'felt_always_sad'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child felt always sad during past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_sad_classif.png') 
plt.close('all')





# Child felt angry or frustrated during past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'felt_angry_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child felt angry or frustrated during past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_angry_classif.png') 
plt.close('all')






# Child felt lonely during past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'felt_lonely_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child felt lonely during past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_lonely_classif.png') 
plt.close('all')






# Child felt lonely during past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'felt_lonely_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child felt lonely during past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_lonely_classif.png') 
plt.close('all')





# Child felt lonely during past week?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'felt_unhappy_cv'] >= 4).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child felt unhappy during past week?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_unhappy_classif.png') 
plt.close('all')







# 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = dfbase__['deaths_per100k'].values
tar_y[np.isnan(tar_y)] = np.nanmean(tar_y)
tar_y = StandardScaler().fit_transform(tar_y[:, None])[:, 0]
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
# folder = StratifiedKFold(shuffle=True, n_splits=10)
# acc = cross_val_score(
#   estimator=lr,
#   X=est.y_scores_, y=tar_y,
#   cv=folder,
#   n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('less deaths vs. more deaths\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Relation to death toll in state since january 2020?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_deaths_per100k_regr.png') 
plt.close('all')



# 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = dfbase__['perc_at_least_one_dose'].values.astype(np.float)
tar_y[np.isnan(tar_y)] = np.nanmean(tar_y)
tar_y = StandardScaler().fit_transform(tar_y[:, None])[:, 0]
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
# folder = StratifiedKFold(shuffle=True, n_splits=10)
# acc = cross_val_score(
#   estimator=lr,
#   X=est.y_scores_, y=tar_y,
#   cv=folder,
#   n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('less vs. more people with 1 dose\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Relation to 1-dose coverage of population?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_perc_at_least_one_dose_regr.png') 
plt.savefig(OUTDIR + f'/boxplot_perc_at_least_one_dose_regr.eps') 
plt.savefig(OUTDIR + f'/boxplot_perc_at_least_one_dose_regr.pdf') 
plt.close('all')






# 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = dfbase__['cases_per100k'].values
tar_y[np.isnan(tar_y)] = np.nanmean(tar_y)
tar_y = StandardScaler().fit_transform(tar_y[:, None])[:, 0]
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
# folder = StratifiedKFold(shuffle=True, n_splits=10)
# acc = cross_val_score(
#   estimator=lr,
#   X=est.y_scores_, y=tar_y,
#   cv=folder,
#   n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer cases vs. more cases\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Relation to number of COVID-19 cases in state since january 2020?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_cases_per100k_regr.png') 
plt.close('all')


y_famhx_ss = dfbase__.famhx_ss.values
# y_famhx_ss = famhx_ss.sum(1)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = StandardScaler().fit_transform(y_famhx_ss[:, None])[:, 0]
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
# folder = StratifiedKFold(shuffle=True, n_splits=10)
# acc = cross_val_score(
#   estimator=lr,
#   X=est.y_scores_, y=tar_y,
#   cv=folder,
#   n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer vs. more issues\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Relation to number of mental health issues in family?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_mental_health_regr2.png') 
plt.close('all')




ple = dfbase__.iloc[:, dfbase__.columns.str.contains('ple_')]
ple = ple.values
ple[ple > 1] = 1
ple[ple < 0] = 0
y_ple = ple.sum(1)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = StandardScaler().fit_transform(y_famhx_ss[:, None])[:, 0]
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
# folder = StratifiedKFold(shuffle=True, n_splits=10)
# acc = cross_val_score(
#   estimator=lr,
#   X=est.y_scores_, y=tar_y,
#   cv=folder,
#   n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer vs. more issues\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Relation to number of mental health issues in family?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_mental_health_regr.png') 
plt.close('all')



#



# Over the past week about how much time per day do you think your child has
# been getting news from television news sources about the coronavirus and its impact? (hours)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LinearRegression(fit_intercept=True) 
# lr = SVC()

tar_y = Y[:, Y_cols == 'child_news_time_cv']
tar_y = StandardScaler().fit_transform(tar_y)[:, 0]
# tar_y = 7 - tar_y  # invert scale: higher number is more hours
# folder = StratifiedKFold(shuffle=True, n_splits=10)
# acc = cross_val_score(
#   estimator=lr,
#   X=est.y_scores_, y=tar_y,
#   cv=folder,
#   n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('fewer vs. more hours\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Amount of COVID19-related TV consumption?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/child_news_time_cv.png') 
plt.savefig(OUTDIR + f'/child_news_time_cv.eps') 
plt.savefig(OUTDIR + f'/child_news_time_cv.pdf') 
plt.close('all')





# What is the most likely news network for this information in your home? 
news_networks = {
1: 'CNN',
2: 'Fox News',
3: 'MSNBC',
4: 'ABC',
5: 'NBC',
6: 'CBS',
7: 'Univision',
8: 'Telemundo',
9: 'PBS/public television PBS/television publica',
10: 'Local news station',
11: 'Comedy News',
12: 'Foreign news station (e.g. BBC)',
13: 'Other news channel',
14: 'No news channel',
15: 'OAN',
16: 'Newsmax'
}

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

for dist_meas, my_title in news_networks.items():

  lr = LogisticRegression(fit_intercept=True) 

  tar_y = Y[:, Y_cols == 'child_news_source_cv'].astype(np.int)
  tar_y = np.squeeze(tar_y == dist_meas).astype(np.int)

  if tar_y.sum() == 0:
    continue

  folder = StratifiedKFold(shuffle=True, n_splits=10)
  acc = cross_val_score(
    estimator=lr,
    X=est.y_scores_, y=tar_y,
    cv=folder,
    n_jobs=4)

  print(acc.mean())
  print(acc.std())

  lr.fit(X=est.y_scores_, y=tar_y)
  print(lr.score(X=est.y_scores_, y=tar_y))
  # 1.0

  bs_coefs = []
  np.random.seed(0)
  for i_bs in range(100):
    bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

    lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
    is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
    print(is_acc)

    print(idx_OOB.sum())
    idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
    oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

    bs_coefs.append(lr.coef_)

  bs_coefs = np.squeeze(bs_coefs)
  plt.figure(figsize=(6, 4))
  plt.boxplot(bs_coefs) 
  plt.xlabel('mode')
  plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
  plt.title(my_title)
  # plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
  plt.tight_layout()
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif.png') 
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif.pdf') 
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif.tiff') 
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif.eps') 

  plt.ylim(-1, +1)
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif_fixedylim.png') 
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif_fixedylim.pdf') 
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif_fixedylim.tiff') 
  plt.savefig(OUTDIR + f'/boxplot_TVchannel_{dist_meas}_classif_fixedylim.eps') 
  plt.close('all')



# child wears mask ?
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'demo_mask_coverage_cv'] >=3).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  # print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Child wears a mask over my face or protective gear ?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_child_wears_mask.png') 
plt.close('all')



# language anchor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import StratifiedKFold, cross_val_score   

lr = LogisticRegression(fit_intercept=True) 

tar_y = np.squeeze(Y[:, Y_cols == 'cv_p_select_language___1']).astype(np.int)
folder = StratifiedKFold(shuffle=True, n_splits=10)
acc = cross_val_score(
  estimator=lr,
  X=est.y_scores_, y=tar_y,
  cv=folder,
  n_jobs=4)

print(acc.mean())
print(acc.std())

lr.fit(X=est.y_scores_, y=tar_y)
print(lr.score(X=est.y_scores_, y=tar_y))
# 1.0

bs_coefs = []
np.random.seed(0)
for i_bs in range(100):
  bs_idx = np.random.randint(0, len(tar_y), len(tar_y))

  lr.fit(est.y_scores_[bs_idx], tar_y[bs_idx])
  is_acc = lr.score(est.y_scores_[bs_idx], tar_y[bs_idx])
  print(is_acc)

  # print(idx_OOB.sum())
  idx_OOB = ~np.in1d(np.arange(len(tar_y)), bs_idx)
  oob_acc = lr.score(est.y_scores_[idx_OOB], tar_y[idx_OOB])

  bs_coefs.append(lr.coef_)

bs_coefs = np.squeeze(bs_coefs)
plt.figure(figsize=(6, 4))
plt.boxplot(bs_coefs) 
plt.xlabel('mode')
plt.ylabel('no vs. yes\n(mean effect +/- bootstrap population intervals)')
plt.title(f'Preferred language is Spanish?\n')
# plt.title('Classifying witnessing of racial discrimination from COVID19 items\nAccuracy in new families: %d.2%%' % (np.mean(oob_acc) * 100))
plt.tight_layout()
plt.savefig(OUTDIR + f'/boxplot_spanish.png') 
plt.close('all')