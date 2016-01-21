package ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment1;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment0.Context;
import ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment0.IntWritable;
import tl.lin.data.pair.PairOfStrings;

public class PairsPMI  extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(PairsPMI.class);

  protected static class CountingWordsMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
	    private static final IntWritable ONE = new IntWritable(1);
	    private final static Text WORD = new Text();
	    private final static String LINE = "*line";
	    
	    @Override
	    public void map(LongWritable key, Text value, Context context)
	        throws IOException, InterruptedException {
	      String line = ((Text) value).toString();
	      StringTokenizer itr = new StringTokenizer(line);

	      int cnt = 0;
	      Set set = Sets.newHashSet();
	      while (itr.hasMoreTokens()) {
	          cnt++;
	          String w = itr.nextToken().toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", "");
	          if (w.length() == 0) continue;
	          set.add(w);
	          WORD.set(w);
	          context.write(WORD, ONE);
	          if (cnt >= 100) break;
	      }
	      WORD.set(LINE);
	      context.write(WORD, ONE);
	    }
	  }
  
  // Reducer: sums up all the counts.
  private static class CountingWordsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    // Reuse objects.
    private final static IntWritable SUM = new IntWritable();

    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      // Sum up values.
      Iterator<IntWritable> iter = values.iterator();
      int sum = 0;
      while (iter.hasNext()) {
        sum += iter.next().get();
      }
      SUM.set(sum);
      context.write(key, SUM);
    }
  }
  
  protected static class MyMapper extends Mapper<LongWritable, Text, PairOfStrings, FloatWritable> {
    private static final FloatWritable ONE = new FloatWritable(1);
    private static final PairOfStrings BIGRAM = new PairOfStrings();

    @Override
    public void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
      String line = ((Text) value).toString();
      StringTokenizer itr = new StringTokenizer(line);

      int cnt = 0;
      Set set = Sets.newHashSet();
      while (itr.hasMoreTokens()) {
          cnt++;
          String w = itr.nextToken().toLowerCase().replaceAll("(^[^a-z]+|[^a-z]+$)", "");
          if (w.length() == 0) continue;
          set.add(w);
          if (cnt >= 100) break;
        }
      
      String[] words = new String[set.size()];
      words = set.toArray(words);
      for (int i = 1; i < tokens.size(); i++) {
        BIGRAM.set(tokens.get(i - 1), tokens.get(i));
        context.write(BIGRAM, ONE);
      }
    }
  }

  protected static class MyCombiner extends
      Reducer<PairOfStrings, FloatWritable, PairOfStrings, FloatWritable> {
    private static final FloatWritable SUM = new FloatWritable();

    @Override
    public void reduce(PairOfStrings key, Iterable<FloatWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      Iterator<FloatWritable> iter = values.iterator();
      while (iter.hasNext()) {
        sum += iter.next().get();
      }
      SUM.set(sum);
      context.write(key, SUM);
    }
  }

  protected static class MyReducer extends
      Reducer<PairOfStrings, FloatWritable, PairOfStrings, FloatWritable> {
    private static final FloatWritable VALUE = new FloatWritable();
    private float marginal = 0.0f;

    @Override
    public void reduce(PairOfStrings key, Iterable<FloatWritable> values, Context context)
        throws IOException, InterruptedException {
      float sum = 0.0f;
      Iterator<FloatWritable> iter = values.iterator();
      while (iter.hasNext()) {
        sum += iter.next().get();
      }

      if (key.getRightElement().equals("*")) {
        VALUE.set(sum);
        context.write(key, VALUE);
        marginal = sum;
      } else {
        VALUE.set(sum / marginal);
        context.write(key, VALUE);
      }
    }
  }

  protected static class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
    @Override
    public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
      return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
    }
  }

  /**
   * Creates an instance of this tool.
   */
  private PairsPMI() {}

  public static class Args {
    @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
    public String input;

    @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
    public String output;

    @Option(name = "-reducers", metaVar = "[num]", required = false, usage = "number of reducers")
    public int numReducers = 1;

    @Option(name = "-textOutput", required = false, usage = "use TextOutputFormat (otherwise, SequenceFileOutputFormat)")
    public boolean textOutput = false;
  }

  /**
   * Runs this tool.
   */
  public int run(String[] argv) throws Exception {
    Args args = new Args();
    CmdLineParser parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(100));

    try {
      parser.parseArgument(argv);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      return -1;
    }

    LOG.info("Tool name: " + PairsPMI.class.getSimpleName());
    LOG.info(" - input path: " + args.input);
    LOG.info(" - output path: " + args.output);
    LOG.info(" - num reducers: " + args.numReducers);
    LOG.info(" - text output: " + args.textOutput);

    Job job = Job.getInstance(getConf());
    job.setJobName(PairsPMI.class.getSimpleName());
    job.setJarByClass(PairsPMI.class);

    job.setNumReduceTasks(args.numReducers);

    FileInputFormat.setInputPaths(job, new Path(args.input));
    FileOutputFormat.setOutputPath(job, new Path(args.output));

//    job.setMapOutputKeyClass(PairOfStrings.class);
//    job.setMapOutputValueClass(FloatWritable.class);
//    job.setOutputKeyClass(PairOfStrings.class);
//    job.setOutputValueClass(FloatWritable.class);
    
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    
    if (args.textOutput) {
      job.setOutputFormatClass(TextOutputFormat.class);
    } else {
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
    }

//    job.setMapperClass(MyMapper.class);
//    job.setCombinerClass(MyCombiner.class);
//    job.setReducerClass(MyReducer.class);
//    job.setPartitionerClass(MyPartitioner.class);
    
    job.setMapperClass(CountingWordsMapper.class);
    job.setCombinerClass(CountingWordsReducer.class);
    job.setReducerClass(CountingWordsReducer.class);

    // Delete the output directory if it exists already.
    Path outputDir = new Path(args.output);
    FileSystem.get(getConf()).delete(outputDir, true);

    long startTime = System.currentTimeMillis();
    job.waitForCompletion(true);
    System.out.println("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

    return 0;
  }

  /**
   * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new PairsPMI(), args);
  }
}
