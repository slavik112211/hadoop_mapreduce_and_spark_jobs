package ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment2

import java.util.StringTokenizer

import scala.collection.JavaConverters._

/*
* Taken from
* https://github.com/lintool/bespin/blob/master/src/main/scala/io/bespin/scala/util/Tokenizer.scala
* */
trait Tokenizer {
  def tokenize(s: String): List[String] = {
    new StringTokenizer(s).asScala.toList
      .map(_.asInstanceOf[String].toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", ""))
      .filter(_.length != 0)
  }
}