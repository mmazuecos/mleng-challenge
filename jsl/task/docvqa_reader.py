from pyspark.sql import DataFrame


class DocVQA():
    """
    Interface class for Scala DocVQAReader.
    """
    def __init__(self):
        pass

    def readDataset(self, spark_session, path):
        """
        Read DocVQA dataset from path.

        Parameters
        ----------
        spark_session : SparkSession
            SparkSession object.
        path : str
            Path to dataset.

        Returns
        -------
        df_pyspark : DataFrame
            PySpark DataFrame object.
        """
        sc = spark_session.sparkContext
        sc.setLogLevel("ERROR")
        doc_vqa_reader = sc._jvm.vqareader.DocVQAReader
        df = doc_vqa_reader.readDataset(spark_session._jsparkSession, path)
        df_pyspark = DataFrame(df, spark_session)

        return df_pyspark
