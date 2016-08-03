package ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment2

import scala.collection.mutable.HashMap

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.Partitioner
import org.rogach.scallop._

class BigramPairsConf(args: Seq[String]) extends ScallopConf(args) with Tokenizer {
  mainOptions = Seq(input, output, reducers)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val reducers = opt[Int](descr = "number of reducers", required = false, default = Some(1))
}

class WordPairFirstWordPartitioner(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    (key.asInstanceOf[List[String]].head.hashCode & Integer.MAX_VALUE) % numPartitions
  }
  // Java equals method to let Spark compare our Partitioner objects
  override def equals(other: Any): Boolean = other match {
    case partitioner: WordPairFirstWordPartitioner =>
      partitioner.numPartitions == numPartitions
    case _ =>
      false
  }
}

//spark-submit --class ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment2.ComputeBigramRelativeFrequencyStripesScala \
//target/bigdata2016w-0.1.0-SNAPSHOT.jar --input data/Shakespeare-test.txt --output shakespeare-stripes-spark --reducers 1
object ComputeBigramRelativeFrequencyPairsScala extends Tokenizer {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new BigramPairsConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())
    log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf()
      .setMaster("local[5]")
      .setAppName("Bigram Count")
      //https://ogirardot.wordpress.com/2015/01/09/changing-sparks-default-java-serialization-to-kryo/
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryoserializer.buffer.mb","24")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val textFile = sc.textFile(args.input())

    val bigrams = textFile
      .flatMap(line => {
        val tokens = tokenize(line)
        if (tokens.length > 1) {
          tokens.sliding(2).toList
        } else List()
      })
      .flatMap(wordPair => {
        List((wordPair -> 1), (List(wordPair.head, "*") -> 1))
      })
      .partitionBy(new WordPairFirstWordPartitioner(3)).persist

    val accum = bigrams.reduceByKey(_+_)


//    }).map(stripe =>{
//      val sum = stripe._2.map(_._2).reduce(_+_)
//      (stripe._1, stripe._2.mapValues(count=>count/sum))
//    })
    accum.saveAsTextFile(args.output()) //.coalesce(1)
  }
}
