import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, silhouette_score, adjusted_rand_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
R={}
data=load_breast_cancer(as_frame=True)
df=data.frame.copy()
df["diagnosis"]=df["target"].map({0:"Malignant",1:"Benign"})
R["shape"]=list(df.shape); R["missing"]=int(df.isna().sum().sum()); R["dups"]=int(df.duplicated().sum())
R["class_counts"]=df["diagnosis"].value_counts().to_dict()
X=df.drop(columns=["target","diagnosis"]); y=df["target"]
# outliers (IQR)
q1,q3=X.quantile(.25),X.quantile(.75); iqr=q3-q1
out=((X<q1-1.5*iqr)|(X>q3+1.5*iqr)).sum().sort_values(ascending=False)
R["outliers_top"]=out.head(5).to_dict()
R["skew_top"]=X.skew().sort_values(ascending=False).head(5).round(2).to_dict()
# EDA plots
sns.set_theme(style="whitegrid")
plt.figure(figsize=(5,3.5)); sns.countplot(x="diagnosis",data=df,palette=["#c0392b","#2e86c1"],hue="diagnosis",legend=False)
plt.title("Class distribution"); plt.tight_layout(); plt.savefig("fig/class.png",dpi=150); plt.close()
fig,ax=plt.subplots(1,3,figsize=(11,3.5))
for a,c in zip(ax,["mean radius","mean concave points","mean texture"]):
    sns.histplot(data=df,x=c,hue="diagnosis",kde=True,ax=a,palette=["#c0392b","#2e86c1"])
plt.tight_layout(); plt.savefig("fig/hist.png",dpi=150); plt.close()
mean_cols=[c for c in X.columns if c.startswith("mean")]
plt.figure(figsize=(8,6.5)); sns.heatmap(X[mean_cols].corr(),annot=True,fmt=".2f",cmap="coolwarm",annot_kws={"size":7})
plt.title("Correlation of mean features"); plt.tight_layout(); plt.savefig("fig/corr.png",dpi=150); plt.close()
corr_t=X.corrwith(y).abs().sort_values(ascending=False).head(6).round(3)
R["corr_target"]=corr_t.to_dict()
plt.figure(figsize=(8,4)); sns.boxplot(data=df,x="diagnosis",y="worst perimeter",palette=["#c0392b","#2e86c1"],hue="diagnosis",legend=False)
plt.title("Worst perimeter by diagnosis"); plt.tight_layout(); plt.savefig("fig/box.png",dpi=150); plt.close()
# supervised
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
models={"Logistic Regression":Pipeline([("s",StandardScaler()),("m",LogisticRegression(max_iter=1000))]),
 "KNN":Pipeline([("s",StandardScaler()),("m",KNeighborsClassifier())]),
 "SVM (RBF)":Pipeline([("s",StandardScaler()),("m",SVC(probability=True,random_state=42))]),
 "Random Forest":RandomForestClassifier(n_estimators=300,random_state=42),
 "Gradient Boosting":GradientBoostingClassifier(random_state=42)}
rows=[]; fitted={}
for n,m in models.items():
    cvs=cross_val_score(m,Xtr,ytr,cv=cv,scoring="f1")
    m.fit(Xtr,ytr); p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]; fitted[n]=(m,pr)
    rows.append(dict(Model=n,CV_F1=round(cvs.mean(),4),CV_std=round(cvs.std(),4),Accuracy=round(accuracy_score(yte,p),4),
      Precision=round(precision_score(yte,p),4),Recall=round(recall_score(yte,p),4),F1=round(f1_score(yte,p),4),AUC=round(roc_auc_score(yte,pr),4)))
res=pd.DataFrame(rows); R["models"]=rows
# tuning
gs=GridSearchCV(Pipeline([("s",StandardScaler()),("m",SVC(probability=True,random_state=42))]),
  {"m__C":[0.1,1,10,100],"m__gamma":["scale",0.01,0.001]},cv=cv,scoring="f1").fit(Xtr,ytr)
best=gs.best_estimator_; p=best.predict(Xte); pr=best.predict_proba(Xte)[:,1]
R["tuned"]=dict(params={k:str(v) for k,v in gs.best_params_.items()},cv_f1=round(gs.best_score_,4),Accuracy=round(accuracy_score(yte,p),4),
  Precision=round(precision_score(yte,p),4),Recall=round(recall_score(yte,p),4),F1=round(f1_score(yte,p),4),AUC=round(roc_auc_score(yte,pr),4))
fitted["Tuned SVM"]=(best,pr)
cm=confusion_matrix(yte,p); R["cm"]=cm.tolist()
plt.figure(figsize=(4.5,3.8)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Malignant","Benign"],yticklabels=["Malignant","Benign"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Tuned SVM confusion matrix"); plt.tight_layout(); plt.savefig("fig/cm.png",dpi=150); plt.close()
plt.figure(figsize=(6,4.5))
for n,(m,pr_) in fitted.items():
    f,t,_=roc_curve(yte,pr_); plt.plot(f,t,label=n)
plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig/roc.png",dpi=150); plt.close()
rf=models["Random Forest"]; imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False).head(10)
R["importance"]=imp.round(4).to_dict()
plt.figure(figsize=(7,4.5)); imp[::-1].plot.barh(color="#2e86c1"); plt.title("Top 10 Random Forest feature importances"); plt.tight_layout(); plt.savefig("fig/imp.png",dpi=150); plt.close()
# unsupervised
Xs=StandardScaler().fit_transform(X)
pca=PCA().fit(Xs); ev=pca.explained_variance_ratio_; R["pca_var"]=[round(float(v),4) for v in ev[:5]]
R["pca_n95"]=int(np.argmax(np.cumsum(ev)>=.95)+1)
sil={}
for k in range(2,8):
    sil[k]=round(silhouette_score(Xs,KMeans(k,n_init=10,random_state=42).fit_predict(Xs)),4)
R["silhouette"]=sil
km=KMeans(2,n_init=10,random_state=42).fit(Xs); lab=km.labels_
ari=adjusted_rand_score(y,lab); R["ari"]=round(ari,4)
ct=pd.crosstab(df["diagnosis"],lab); R["crosstab"]=ct.values.tolist()
Z=PCA(2).fit_transform(Xs)
fig,ax=plt.subplots(1,3,figsize=(14,4))
ax[0].plot(range(1,len(ev)+1),np.cumsum(ev),marker="o",ms=3); ax[0].axhline(.95,color="r",ls="--"); ax[0].set_title("PCA cumulative variance"); ax[0].set_xlabel("Components")
ax[1].plot(list(sil.keys()),list(sil.values()),marker="o"); ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k")
ax[2].scatter(Z[:,0],Z[:,1],c=lab,cmap="coolwarm",s=12); ax[2].set_title("K-Means (k=2) on PCA space")
plt.tight_layout(); plt.savefig("fig/unsup.png",dpi=150); plt.close()
plt.figure(figsize=(5.5,4.5)); 
for c,col in zip(["Malignant","Benign"],["#c0392b","#2e86c1"]):
    m=df["diagnosis"]==c; plt.scatter(Z[m,0],Z[m,1],s=12,c=col,label=c)
plt.legend(); plt.title("True diagnosis in PCA space"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig("fig/pca_true.png",dpi=150); plt.close()
