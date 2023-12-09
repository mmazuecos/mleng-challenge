package vqareader

import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    // Log that we are starting the program
    println("Starting DocVQAReaderApp")
    val spark = SparkSession
        .builder()
        .appName("DocVQAReader App")
        .config("spark.master", "local")
        .getOrCreate()


    val datasetPath = "/Users/maurygreen/assignment-jsl-ml-engineer/data/DocVQA/train_v1.0_withQT.json"
    val imagesPath = "path/to/images"
    // Log that we are about to read the dataset
    println(s"Reading dataset from $datasetPath")
    val dataFrame = DocVQAReader.readDataset(spark, datasetPath)

    dataFrame.show(1, false) // To display the contents of the DataFrame
    
    spark.stop()
  }
}
