package ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment2

import scala.collection.mutable.HashMap

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._

class BigramConf(args: Seq[String]) extends ScallopConf(args) with Tokenizer {
  mainOptions = Seq(input, output, reducers)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val reducers = opt[Int](descr = "number of reducers", required = false, default = Some(1))
}

//spark-submit --class ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment2.ComputeBigramRelativeFrequencyStripesScala \
//target/bigdata2016w-0.1.0-SNAPSHOT.jar --input data/Shakespeare-test.txt --output shakespeare-stripes-spark --reducers 1
object ComputeBigramRelativeFrequencyStripesScala extends Tokenizer {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new BigramConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())
    log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf().setAppName("Bigram Count")
//    conf.setMaster("local[5]")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val textFile = sc.textFile(args.input())

    val bigrams = textFile
      .flatMap(line => {
        val stripes = new HashMap[String, HashMap[String, Float]]
        val tokens = tokenize(line)
        if (tokens.length > 1) {
          val pairsIter = tokens.sliding(2)
          while(pairsIter.hasNext) {
            val wordPair = pairsIter.next
            stripes.get(wordPair.head) match {
              case Some(stripe) => {
                stripe.get(wordPair.last) match {
                  case Some(prevCount) => {
                    stripe.put(wordPair.last, prevCount+1)
                  } case None => {
                    stripe.put(wordPair.last, 1)
                  }
                }
              } case None => {
                val stripe = new HashMap[String, Float]
                stripe.put(wordPair.last, 1)
                stripes.put(wordPair.head, stripe)
              }
            }
          }
        }
        stripes
      }).reduceByKey((stripe1, stripe2) => {
      (stripe1++stripe2.map{ case (k,v) => k -> (v+stripe1.getOrElse(k,0.0f)) })
        .asInstanceOf[HashMap[String,Float]]
      }).map(stripe =>{
        val sum = stripe._2.map(_._2).reduce(_+_)
        (stripe._1, stripe._2.mapValues(count=>count/sum))
      })
    bigrams.coalesce(1).saveAsTextFile(args.output())
  }
}
