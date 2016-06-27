package ca.uwaterloo.cs.bigdata2016w.slavik112211.assignment1;

import java.io.IOException;
import java.net.URI;
import java.util.*;

import com.google.common.collect.Sets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import tl.lin.data.map.HMapStIW;
import tl.lin.data.pair.PairOfStrings;

/*
* MapReduce job to calculate Pointwise mutual information of 2 words PMI(x,y) in a text corpora.
* Kenneth W. Church and Patrick Hanks. Word association norms, mutual information, and lexicography, 1990
* Calculation consists of 3 steps: map - reduce - map.
* See assignments.html #1 for details on the task.
* See assignment1.odt for details on solution.
* */
public class StripesPMI extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(StripesPMI.class);
  enum Lines { TOTAL_COUNT }
  private static final String PLACEHOLDER_WORD = "*";
  private static final Text   PLACEHOLDER_WORD_WRITABLE = new Text(PLACEHOLDER_WORD);
  private static final String firstJobOutputPath = "./data/firstjobOutput/";
  private static final String outputPathWordsCounts = "wordsCounts/output";
  private static final String inputPathWordsCounts = "wordsCounts/";
  private static final String outputPathWordPairsCounts = "wordPairsCounts/output";
  private static final String inputPathWordPairsCounts = "wordPairsCounts/";

  protected static class UniqueWordsAndWordPairsCounterMapper extends Mapper<LongWritable, Text, Text, HMapStIW> {
    private static final int ONE = 1;
    private static final Text WORD_KEY = new Text();
    private static final HMapStIW MAP = new HMapStIW();
    String[] uniqueWords;

    public static String[] extractUniqueWordsFromInputLine(Text value){
      String line = value.toString();
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
      words = (String[]) set.toArray(words);
      return words;
    }

    @Override
    public void map(LongWritable inputLineKey, Text inputLine, Context context)
        throws IOException, InterruptedException {
      uniqueWords = this.extractUniqueWordsFromInputLine(inputLine);
      MAP.clear();

      // 1. Emit encountered single words to reducer. This is needed to calculate
      // Pr[word x appears in a line]=#[word x appeared in a line]/#[total input lines]
      // these Pr(x) are used in PMI(x,y) formula denominator.
      for (String word : uniqueWords) {
        MAP.put(word, ONE);
      }
      context.write(PLACEHOLDER_WORD_WRITABLE, MAP);
      MAP.clear();

      // 2. Emit encountered word pairs to reducer. This is needed to calculate
      // Pr[words x and y appear in a line]=#[words x and y appeared in a line]/#[total input lines]
      // these Pr(x,y) are used in PMI(x,y) formula nominator
      if (uniqueWords.length > 1) {
        for (int i = 0; i < uniqueWords.length; i++) {
          for (int j = 0; j < uniqueWords.length; j++) {
            if(i==j) continue;
            MAP.put(uniqueWords[j], ONE);
          }
          WORD_KEY.set(uniqueWords[i]);
          context.write(WORD_KEY, MAP);
          MAP.clear();
        }
      }

      // 3. Account this inputLine to calculate #[total input lines] in reducer
      context.getCounter(Lines.TOTAL_COUNT).increment(1);
    }
  }

  private static class UniqueWordsAndWordPairsCounterCombiner
          extends Reducer<Text, HMapStIW, Text, HMapStIW> {
    private static final HMapStIW MAP = new HMapStIW();

    @Override
    public void reduce(Text wordLHS, Iterable<HMapStIW> wordsPairPartialCountList, Context context)
            throws IOException, InterruptedException {
      Iterator<HMapStIW> iter = wordsPairPartialCountList.iterator();
      MAP.clear();
      String wordRHS;
      while (iter.hasNext()) {
        HMapStIW currentMap = iter.next();
        Iterator<String> currentMapIter = currentMap.keySet().iterator();
        while(currentMapIter.hasNext()){
          wordRHS = currentMapIter.next(); //Right-hand side word in a word pair
          MAP.increment(wordRHS, currentMap.get(wordRHS));
        }
      }
      context.write(wordLHS, MAP);
    }
  }

  private static class UniqueWordsAndWordPairsCounterReducer
          extends Reducer<Text, HMapStIW, Text, HMapStIW> {
    private static final HMapStIW MAP = new HMapStIW();
    public static final String firstJobOutputChannel = "firstJobOutput";

    private MultipleOutputs<Text, HMapStIW> multipleOutputs;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      multipleOutputs = new MultipleOutputs<Text, HMapStIW>(context);
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      multipleOutputs.close();
    }

    /**
     * If first element in word pair is a PLACEHOLDER_WORD,
     * then it's not a pair, but rather a single-word occurrence counter.
     * We put this single-word counters aside into separate files.
     * In the actual PMI(x,y) calculation in second map job these files
     * are pulled up as a distributed cache
     */
    protected String getOutputPathForKey(Text word) {
      return (word.equals(PLACEHOLDER_WORD_WRITABLE) ?
              outputPathWordsCounts : outputPathWordPairsCounts);
    }

    @Override
    public void reduce(Text wordLHS, Iterable<HMapStIW> wordsPairPartialCountList, Context context)
        throws IOException, InterruptedException {
      Iterator<HMapStIW> iter = wordsPairPartialCountList.iterator();
      MAP.clear();
      String wordRHS;
      while (iter.hasNext()) {
        HMapStIW currentMap = iter.next();
        Iterator<String> currentMapIter = currentMap.keySet().iterator();
        while(currentMapIter.hasNext()){
          wordRHS = currentMapIter.next(); //Right-hand side word in a word pair
          MAP.increment(wordRHS, currentMap.get(wordRHS));
        }
      }

      // To reduce the number of spurious pairs, we are only interested
      // in pairs of words that co-occur in ten or more lines.
      Iterator<String> mapIter = MAP.keySet().iterator();
      while(mapIter.hasNext()){
        wordRHS = mapIter.next(); //Right-hand side word in a word pair
        if(!wordLHS.equals(PLACEHOLDER_WORD) && MAP.get(wordRHS)<10) mapIter.remove();
      }
      multipleOutputs.write(firstJobOutputChannel, wordLHS, MAP, getOutputPathForKey(wordLHS));
    }
  }

  public static class FloatArrayWritable extends ArrayWritable {
    public FloatArrayWritable() { super(FloatWritable.class); }

    public String toString(){
      String stringValue="";
      Writable[] valuesArray = get();
      for(int i=0; i<valuesArray.length; i++){
        stringValue+=String.format("%.5f", ((FloatWritable) valuesArray[i]).get());
        if(i!=valuesArray.length-1) stringValue+=", ";
      }
      return stringValue;
    }
  }

  protected static class PointwiseMutualInformationMapper extends Mapper<PairOfStrings, IntWritable, PairOfStrings, FloatArrayWritable> {
    private long linesTotalCount;
    Path[] localFiles;
    private static final FloatWritable
      countWordA   = new FloatWritable(), probabilityWordA  = new FloatWritable(),
      countWordB   = new FloatWritable(), probabilityWordB  = new FloatWritable(),
      countWordsAB = new FloatWritable(), probabilityWordsAB = new FloatWritable(),
      wordsAB_PMI  = new FloatWritable(), wordsAB_PMIlog10 = new FloatWritable();
    private static final FloatArrayWritable wordsPairPMIArray = new FloatArrayWritable();
    private static final FloatWritable[] wordsPairPMIValues = new FloatWritable[]{
      countWordA, probabilityWordA, countWordB, probabilityWordB,
      countWordsAB, probabilityWordsAB, wordsAB_PMI, wordsAB_PMIlog10};
//    private static final FloatWritable[] wordsPairPMIValues = new FloatWritable[]{wordsAB_PMIlog10};
    private HashMap<String, Integer> dictionaryWordCount = new HashMap<String, Integer>(30000); //Shakespeare's dictionary size

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      linesTotalCount = conf.getLong("linesTotalCount", 1);
      LOG.info("Job2. linesTotalCount: " + linesTotalCount);

      //http://stackoverflow.com/questions/24647992/wildcard-in-hadoops-filesystem-listing-api-calls
//      FileSystem fs = FileSystem.get(URI.create("./wordsCounts/output-r-00000"), conf);
//      context.getFile
//      RemoteIterator<LocatedFileStatus> files = filesystem.listFiles(new Path("./wordsCounts/output-r-00000"), true);
//      Pattern pattern = Pattern.compile("^.*/date=[0-9]{8}/A-schema\\.avsc$");
//      while (files.hasNext()) {
//        Path path = files.next().getPath();
//        if (pattern.matcher(path.toString()).matches())
//        {
//          System.out.println(path);
//        }
//      }
      localFiles = DistributedCache.getLocalCacheFiles(conf);
      Path dictionaryCount = new Path(firstJobOutputPath+inputPathWordsCounts+"/output-r-00000");
      SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(dictionaryCount));
      PairOfStrings key = new PairOfStrings();
      IntWritable val = new IntWritable();
      while (reader.next(key, val)) {
        dictionaryWordCount.put(key.getRightElement(), val.get());
      }
      reader.close();
      LOG.info("Job2. dictionary loaded in memory. Size: " + dictionaryWordCount.size());
    }

    @Override
    public void map(PairOfStrings wordsPair, IntWritable count, Context context)
            throws IOException, InterruptedException {
      countWordsAB.set((float) count.get());
      probabilityWordsAB.set(((float) count.get())/linesTotalCount);

      countWordA.set((float) dictionaryWordCount.get(wordsPair.getLeftElement()));
      probabilityWordA.set(countWordA.get()/linesTotalCount);

      countWordB.set((float) dictionaryWordCount.get(wordsPair.getRightElement()));
      probabilityWordB.set(countWordB.get()/linesTotalCount);

      //Calc 101 ;) Parenthesis are mandatory: 1a) 4/4*2=2; 2a) 4/2*4=8; whereas 1b) 4/(4*2)=0.5; 2b) 4/(2*4)=0.5
      wordsAB_PMI.set(probabilityWordsAB.get()/(probabilityWordA.get()*probabilityWordB.get()));
      wordsAB_PMIlog10.set((float) Math.log10((double) wordsAB_PMI.get()));
      wordsPairPMIArray.set(wordsPairPMIValues);
      context.write(wordsPair, wordsPairPMIArray);
    }
  }

  /**
   * Creates an instance of this tool.
   */
  private StripesPMI() {}

  public static class Args {
    @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
    public String input;

    @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
    public String output;

    @Option(name = "-reducers", metaVar = "[num]", required = false, usage = "number of reducers")
    public int numReducers = 1;
  }

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

    LOG.info("Tool name: " + StripesPMI.class.getSimpleName());
    LOG.info(" - input path: " + args.input);
    LOG.info(" - output path: " + args.output);
    LOG.info(" - num reducers: " + args.numReducers);

    Configuration jobConfig = getConf();

    Job job = Job.getInstance(jobConfig);
    job.setJobName(StripesPMI.class.getSimpleName());
    job.setJarByClass(StripesPMI.class);

    job.setNumReduceTasks(args.numReducers);
    MultipleOutputs.addNamedOutput(job, UniqueWordsAndWordPairsCounterReducer.firstJobOutputChannel,
        SequenceFileOutputFormat.class, Text.class, HMapStIW.class);
    FileInputFormat.setInputPaths(job, new Path(args.input));
    FileOutputFormat.setOutputPath(job, new Path(firstJobOutputPath));

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(HMapStIW.class);

    job.setMapperClass(UniqueWordsAndWordPairsCounterMapper.class);
    job.setCombinerClass(UniqueWordsAndWordPairsCounterCombiner.class);
    job.setReducerClass(UniqueWordsAndWordPairsCounterReducer.class);

    // Delete the output directory if it exists already.
    Path outputDir = new Path(firstJobOutputPath);
    FileSystem.get(jobConfig).delete(outputDir, true);

    long startTime = System.currentTimeMillis();
    job.waitForCompletion(true);

    jobConfig.setLong("linesTotalCount", job.getCounters().findCounter(Lines.TOTAL_COUNT).getValue());
    System.out.println("First job finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

    // --------------------------------------------------
    // 2nd MapReduce job. Only has a mapper, no reduce step.
    // Calculation of PMI(x,y) values based on the uniqueWord and uniqueWordPairs counts received in first MR job
    DistributedCache.addCacheFile(new URI(firstJobOutputPath+inputPathWordsCounts+"/output-r-00000"), jobConfig);
//    Job job2 = Job.getInstance(jobConfig);
//    job2.setJobName(StripesPMI.class.getSimpleName());
//    job2.setJarByClass(StripesPMI.class);
//
//    job2.setNumReduceTasks(0);
//    job2.setInputFormatClass(SequenceFileInputFormat.class);
//    job2.setOutputFormatClass(TextOutputFormat.class);
//    FileInputFormat.setInputPaths(job2, new Path(firstJobOutputPath+inputPathWordPairsCounts));
//    FileOutputFormat.setOutputPath(job2, new Path(args.output));
//
//    job2.setMapOutputKeyClass(PairOfStrings.class);
//    job2.setMapOutputValueClass(FloatArrayWritable.class);
//    job2.setMapperClass(PointwiseMutualInformationMapper.class);
//
//    // Delete the output directory if it exists already.
//    Path outputDir2 = new Path(args.output);
//    FileSystem.get(jobConfig).delete(outputDir2, true);
//
//    long startTime2 = System.currentTimeMillis();
//    job2.waitForCompletion(true);
//    System.out.println("Second job finished in " + (System.currentTimeMillis() - startTime2) / 1000.0 + " seconds");

    return 0;
  }

  /**
   * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new StripesPMI(), args);
  }
}
