package com.test.demo;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class Tester {


    public static final String DATASETPATH = "/home/amin/Documents/Rajman/demo1/src/main/resources/model.arff";
    public static final String MODElPATH = "/home/amin/Documents/Rajman/demo1/src/main/resources/generated-model.arff";

    public static void main(String[] args) throws Exception {

        ModelGenerator mg = new ModelGenerator();

        Instances dataset = mg.loadDataset(DATASETPATH);

        Filter filter = new Normalize();

        int trainSize = (int) Math.round(dataset.numInstances() * 0.9);
        int testSize = dataset.numInstances() - trainSize;

        dataset.randomize(new Debug.Random(1));

        filter.setInputFormat(dataset);
        Instances datasetnor = Filter.useFilter(dataset, filter);

        Instances traindataset = new Instances(datasetnor, 0, trainSize);

        MultilayerPerceptron ann = (MultilayerPerceptron) mg.buildClassifier(traindataset);


        mg.saveModel(ann, MODElPATH);
        ModelClassifier cls = new ModelClassifier();


        String classname =cls.classifiy(Filter.useFilter(cls.createInstance("overcast","mild","normal","TRUE"), filter), MODElPATH);
        System.out.println("\n The class name for the result is  " +classname);

    }
}
