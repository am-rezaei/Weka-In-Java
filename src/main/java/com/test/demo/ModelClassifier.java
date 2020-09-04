package com.test.demo;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class ModelClassifier {

//    @attribute outlook {sunny, overcast, rainy}
//    @attribute temperature {hot, mild, cool}
//    @attribute humidity {high, normal}
//    @attribute windy {TRUE, FALSE}
//    @attribute play {yes, no}


    private ArrayList attributes;
    private Instances dataRaw;

    Attribute outlook;
    Attribute temperature;
    Attribute humidity;
    Attribute windy;
    Attribute play;

    public ModelClassifier() {
        attributes = new ArrayList();
        outlook = addAttribute("outlook", new String[]{"sunny", "overcast", "rainy"});
        temperature = addAttribute("temperature", new String[]{"hot", "mild", "cool"});
        humidity = addAttribute("humidity", new String[]{"high", "normal"});
        windy = addAttribute("windy", new String[]{"TRUE", "FALSE"});
        play = addAttribute("play", new String[]{"yes", "no"});
        dataRaw = new Instances("TestInstances", attributes, 0);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
    }


    private Attribute addAttribute(String attributeName, Object[] values) {
        ArrayList classVal = new ArrayList();
        Arrays.asList(values).stream().forEach(p -> {
            classVal.add(p);
        });
        Attribute result = new Attribute(attributeName, classVal);
        attributes.add(result);
        return result;
    }


    public Instances createInstance(String outlookStr, String temperatureStr, String humidityStr, String windyStr) {
        dataRaw.clear();

        DenseInstance instance = new DenseInstance(5);
        instance.setValue(outlook, outlookStr);
        instance.setValue(temperature, temperatureStr);
        instance.setValue(humidity, humidityStr);
        instance.setValue(windy, windyStr);
        dataRaw.add(instance);

        return dataRaw;
    }


    public String classifiy(Instances insts, String path) {
        String result = "Not classified!!";
        Classifier cls = null;
        try {
            cls = (MultilayerPerceptron) SerializationHelper.read(path);
            result = (String) Arrays.asList(new String[]{"yes", "no"}).get((int) cls.classifyInstance(insts.firstInstance()));
        } catch (Exception ex) {
            Logger.getLogger(ModelClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }


    public Instances getInstance() {
        return dataRaw;
    }


}
