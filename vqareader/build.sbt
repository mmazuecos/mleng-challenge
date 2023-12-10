name := "DocVQAReader"

version := "0.1"

scalaVersion := "2.12.18" // Use a version compatible with your Spark version

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.1", // Use the Spark version compatible with your environment
  "org.apache.spark" %% "spark-sql" % "3.1.1",
  "net.sf.py4j" % "py4j" % "0.10.9" // Add Py4J dependency
  // Add any other dependencies you need
)
