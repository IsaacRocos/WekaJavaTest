/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pruebaweka;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

/**
 *
 * @author Isaac
 */
public class PruebaWEKA {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("ventasPrac2.arff"));
        
        Instances train = new Instances(breader); //Read all the instances in the file (ARFF, CSV, XRFF, ...
        train.setClassIndex(train.numAttributes() -1); 
        
        breader.close();
        
        //NaiveBayes nB = new NaiveBayes();
        //nb.buildClassifier(train);
        
        System.out.println(train.toSummaryString());
        
        ZeroR zr = new ZeroR();
        zr.buildClassifier(train);
        
        System.out.println(zr.globalInfo());
        
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(zr, train, 10,new Random(1));        
        System.out.println(eval.toSummaryString("=RESULTADOS=",true)); //resultados
        //System.out.println(eval.fMeasure(1) + "" + eval.precision(1) + " " + eval.recall(1)); //
        
    }
    
}
