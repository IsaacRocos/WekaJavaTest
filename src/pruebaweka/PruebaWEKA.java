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
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SimpleLinearRegression;
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
        
        //Probando ZeroR
        System.out.println(train.toSummaryString());
        System.out.println("");
        ZeroR zr = new ZeroR();
        zr.buildClassifier(train);
        System.out.println(zr.globalInfo());        
        new PruebaWEKA().evaluar(train,zr);
        
        
        // Probando regresion lineal simple
        SimpleLinearRegression slr = new SimpleLinearRegression();
        slr.buildClassifier(train);
        new PruebaWEKA().evaluar(train,slr);
        
    }
    
    
    public void evaluar(Instances train, AbstractClassifier clasificador) throws Exception{
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(clasificador, train, 10,new Random(1));        
        System.out.println(eval.toSummaryString("=RESULTADOS <" + clasificador.getClass().getSimpleName() + ">=",true)); //resultados
        //System.out.println(eval.fMeasure(1) + "" + eval.precision(1) + " " + eval.recall(1)); //
        
    }
    
}
