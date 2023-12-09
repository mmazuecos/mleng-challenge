import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object DocVQAReader {
  def readDataset(spark: SparkSession, path: String): DataFrame = {
    // Define the schema based on the dataset's structure
    val schema = new StructType()
      .add("path", StringType)
      .add("modificationTime", TimestampType)
      .add("questions", ArrayType(StringType))
      .add("answers", ArrayType(ArrayType(StringType)))
      // Add any other fields you need

    // Read the JSON dataset
    val df = spark.read.schema(schema).json(path)

    // Load and add image data as binary
    // This might involve reading image files and adding them to the DataFrame

    // Return the final DataFrame
    df
  }
}
