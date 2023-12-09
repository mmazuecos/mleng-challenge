package vqareader

import org.apache.spark.sql.types._

object SchemaDefinition {
  // Input schema of the DocVQA dataset
  val docVQASchema: StructType = new StructType()
    .add("questionId", IntegerType)
    .add("question", StringType)
    .add("question_types", ArrayType(StringType))
    .add("image", StringType)
    .add("docId", IntegerType)
    .add("ucsf_document_id", StringType)
    .add("ucsf_document_page_no", StringType)
    .add("answers", ArrayType(StringType))
    .add("data_split", StringType)

  // Output schema of the DocVQA dataset
  val docVQASchemaOutput: StructType = new StructType()
    .add("path", StringType, nullable = true)
    .add("modificationTime", TimestampType, nullable = true)
    .add("length", LongType, nullable = true)
    .add("content", BinaryType, nullable = true)
    .add("questions", ArrayType(StringType, containsNull = true), nullable = true)
    .add("answers", ArrayType(ArrayType(StringType, containsNull = true), containsNull = true), nullable = true)
}