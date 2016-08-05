package ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment2

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

case class WordPairKey(word1: String, word2: String) extends Ordered[WordPairKey] {
  // Returns `x` where:
  // `x <  0` when `this <  that`
  // `x == 0` when `this == that`
  // `x >  0` when `this >  that`
  def compare(that: WordPairKey): Int = {
    if(this.word1 != that.word1)
      this.word1 compare that.word1
    else if(this.word2 != "*" && that.word2 != "*")
      this.word2 compare that.word2
    else if(this.word2 == "*" && that.word2 != "*") -1
    else if(this.word2 != "*" && that.word2 == "*") 1
    else  0 //this.word2 == "*" && that.word2 == "*"
  }
}

class WordPairFirstWordPartitioner(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    (key.asInstanceOf[WordPairKey].word1.hashCode & Integer.MAX_VALUE) % numPartitions
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
//      .setMaster("local[5]")
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
        val bigrams = if (tokens.length > 1) tokens.sliding(2).toList else List()
        bigrams.flatMap(wordPair => List(WordPairKey(wordPair.head, wordPair.last), WordPairKey(wordPair.head, "*")))
          .map(wordPair => wordPair->1)
      })

    //reduceByKey() does a reshuffle of data across partitions,
    //and then right after it, repartitionAndSortWithinPartitions() also does a reshuffle of data across partitions ;(
    //ideally there would be one reshuffle.
    //repartitioning is needed to ensure that all wordPairs with same word1 will end up in same partitions.
    //sorting is needed to put ("foo", "*") on top of all other ("foo", "bar") pairs.
    //("foo", "*") represents accumulated count of all pairs ("foo", "bar"), ("foo", "baz"), ("foo", "qux")

    val bigramsAggregated = bigrams
      .reduceByKey(_+_)
      .repartitionAndSortWithinPartitions(new WordPairFirstWordPartitioner(args.reducers())).persist

    val bigramsWeight = bigramsAggregated.mapPartitions(iter =>{
      var outputList = List[(WordPairKey, Float)]()
      //I'd better init totalPerWordPair to something else before dividing,
      //unless of course I want the galaxy to explode ;)
      var totalPerWordPair: Float = 0
      while(iter.hasNext){
        val wordPair = iter.next
        if(wordPair._1.word2=="*")
          totalPerWordPair = wordPair._2.toFloat
        else
          outputList .::= (wordPair._1, wordPair._2/totalPerWordPair)
      }
      outputList.iterator
    }, true) //true - preserve partitions

//    bigramsAggregated.saveAsTextFile(args.output()+"2")
    bigramsWeight.saveAsTextFile(args.output())
  }
}
