package vqareader

import org.apache.log4j.Logger
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import java.nio.file.{Files, Paths}
import scala.util.Try
import py4j.GatewayServer;


object DocVQAReader extends App{

  // Initialize the logger
  val logger = Logger.getLogger(DocVQAReader.getClass)

  def loadImage2(path: String, thing: String): Array[Byte] = {
    Try(Files.readAllBytes(Paths.get(path))).getOrElse(Array[Byte]())
  }

  def show_me(): Unit = {
    println("Hello from Scala!")
  }

  def getDirectoryFromPath(filePath: String): String = {
    new java.io.File(filePath).getParent
  }

  def readDataset(spark: SparkSession, path: String): DataFrame = {
    // Not the best, just a workaround
    val loadImage2UDF = udf(loadImage2 _)
    spark.udf.register("loadImage2UDF", loadImage2UDF)

    val directory = getDirectoryFromPath(path)

    // Read the JSON file
    val rawDf = spark.read.json(path)

    // Check if certain columns exist
    val hasAnswers = rawDf.columns.contains("answers")
    val hasQuestionTypes = rawDf.columns.contains("question_types")

    // Assuming the 'data' field contains an array of records, you'll need to explode this array
    val explodedDf = rawDf.select(explode(col("data")))

    // Select and rename nested fields to match your schema
    val df = explodedDf.select(
      col("col.questionId"),
      col("col.question"),
      if (hasQuestionTypes) col("col.question_types") else lit(null).alias("question_types"),
      col("col.image"),
      col("col.docId"),
      if (hasAnswers) col("col.answers") else lit(null).alias("answers"),
      col("col.data_split")
    ).filter(_ != null)

    // Group by 'docId' and aggregate other fields
    val groupedDf = df.groupBy("docId")
      .agg(
        collect_list("question").alias("questions"),
        collect_list("answers").alias("answers"),
        first("image").alias("path")
      )
      .withColumn("content", loadImage2UDF(concat_ws("/", lit(directory), col("path")), lit("")))
      .withColumn("length", size(col("questions"))) // Add the length of the question list
      .withColumn("modificationTime", current_timestamp()) // Add the timestamp of the current time

    // Select columns in the order specified in the schema
    groupedDf.select(
      col("path"),
      col("modificationTime"),
      col("length"),
      col("content"),
      col("questions"),
      col("answers")
    )
  }
}
