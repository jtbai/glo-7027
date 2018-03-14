if CONFIGS.use_linear_model:

if CONFIGS.use_generalised_linear_model:
    print('GLM')
    start_time = time.time()
    regressionGLM(X_train, y_train, X_test, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')

if CONFIGS.use_generalised_additive_model:
    print('GAM')
    start_time = time.time()
    regressionGAM(X_train, y_train, X_test, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')

if CONFIGS.use_gradient_boosting:
    print('Gradient Boosting')
    start_time = time.time()
    regressionGradientBoosting(X_train, y_train, X_test, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')

if CONFIGS.use_svm:
    print('SVM')
    start_time = time.time()
    regressionSVM(X_train, y_train, X_test, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')

if CONFIGS.use_random_forest:
    print('Random forest')
    start_time = time.time()
    regressionRandomForest(X_train, y_train, X_test, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')

print('Total')
print("--- %s seconds ---" % (time.time() - start_time_global))
