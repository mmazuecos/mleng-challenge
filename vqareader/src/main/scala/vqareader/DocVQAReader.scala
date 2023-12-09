package vqareader

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import java.nio.file.{Files, Paths}

object DocVQAReader {

  // Define the UDF for loading images
  val loadImage = udf((path: String) => {
    try {
      Files.readAllBytes(Paths.get(path))
    } catch {
      case e: Exception => Array[Byte]() // Return empty byte array in case of exception
    }
  })

  private def getDirectoryFromPath(filePath: String): String = {
    new java.io.File(filePath).getParent
  }

  def readDataset(spark: SparkSession, path: String): DataFrame = {
    val directory = getDirectoryFromPath(path)

    // Read the JSON file
    val rawDf = spark.read.json(path)

    // Assuming the 'data' field contains an array of records, you'll need to explode this array
    val explodedDf = rawDf.select(explode(col("data")))

    // Select and rename nested fields to match your schema
    val df = explodedDf.select(
      col("col.questionId"),
      col("col.question"),
      col("col.question_types"),
      col("col.image"),
      col("col.docId"),
      col("col.answers"),
      col("col.data_split")
    )

    // Group by 'docId' and aggregate other fields
    val groupedDf = df.groupBy("docId")
      .agg(
        collect_list("question").alias("questions"),
        collect_list("answers").alias("answers"),
        first("image").alias("path")
      )
      .withColumn("content", loadImage(concat(lit(directory), lit("/"), col("path"))))
      .withColumn("length", size(col("questions"))) // Add the length of the question list
      .withColumn("modificationTime", current_timestamp()) // Add the timestamp of the current time

    groupedDf
  }
}
