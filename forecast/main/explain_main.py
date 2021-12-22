from sklearn.utils import all_estimators

estimators = all_estimators(type_filter='classifier')

all_clfs = []
all_shap_dicts = dict()
for name, ClassifierClass in estimators:
    try:
        clf = ClassifierClass()
        clf.fit(X_train, y_train)
        all_shap_dicts[name] = gen_dict_mean_shap_no_plot(clf,X_train,X_test,name)
        all_clfs.append(clf)
        print(name)
    except Exception as e:
        print(e)
        continue