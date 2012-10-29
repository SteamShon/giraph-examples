package com.skplanet.ydy.giraph.examples.lbp;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.giraph.graph.BspUtils;
import org.apache.giraph.graph.Edge;
import org.apache.giraph.graph.EdgeListVertex;
import org.apache.giraph.graph.GiraphJob;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.graph.VertexReader;
import org.apache.giraph.graph.VertexWriter;
import org.apache.giraph.lib.TextVertexInputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat.TextVertexWriter;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Maps;
/**
 * This is implementation for "http://wwwconference.org/www2008/papers/pdf/p61-fuxmanA.pdf"
 * 
 * @author ydy
 *
 */
public class RandomWalkVertex extends EdgeListVertex<LongWritable, VectorWritable, FloatWritable, MultiLabelVectorWritable> implements Tool{
  public static final String MIN_NON_CONCEPT_VERTEX_ID = RandomWalkVertex.class.getName() + ".minNonConceptVertexId";
  public static final String GAMMA = RandomWalkVertex.class.getName() + ".gamma";
  public static final String ITERATION = RandomWalkVertex.class.getName() + ".iteration";
  
  private int minNonConceptVertedId = 25425;
  
  public static double gamma = 1e-6;
  public static int iteration = 5;
  
  
  @Override
  public void initialize(LongWritable id, VectorWritable value,
      Map<LongWritable, FloatWritable> edges,
      Iterable<MultiLabelVectorWritable> messages) {
    super.initialize(id, value, edges, messages);
    this.minNonConceptVertedId = getConf().getInt(MIN_NON_CONCEPT_VERTEX_ID, 0);
    gamma = getConf().getFloat(GAMMA, 1e-10f);
    iteration = getConf().getInt(ITERATION, 10);
  }
  
  @Override
  public void compute(Iterable<MultiLabelVectorWritable> messages) throws IOException {
    // assumes that we have probabilities for this vertex per concept
    long step = getSuperstep();
    if (step < iteration) {
      
      Vector currentVector = getValue().get();
      System.out.println(getId() + "\tCurrent Vector: " + currentVector);
      
      MultiLabelVectorWritable newMessage = new MultiLabelVectorWritable();
      newMessage.setLabels(new int[]{(int)getId().get()});
      Vector newMessageVector = new RandomAccessSparseVector(this.minNonConceptVertedId);
      
      for (MultiLabelVectorWritable message : messages) {
        int messageId = message.getLabels()[0];
        Vector conceptProbs = message.getVector();
        float weight = getEdgeValue(new LongWritable(messageId)).get();
        Iterator<Vector.Element> probs = conceptProbs.iterateNonZero();
        while (probs.hasNext()) {
          Vector.Element prob = probs.next();
          int conceptId = prob.index();
          currentVector.set(conceptId, Math.max(currentVector.get(conceptId),  prob.get() * weight)); 
        }
      }
      // prunning
      
      Iterator<Vector.Element> iter = currentVector.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element e = iter.next();
        if (e.get() < gamma) {
          continue;
        }
        newMessageVector.setQuick(e.index(), e.get());
      }
      newMessage.setVector(newMessageVector);
      System.out.println(getId() + "\tNext Vector: " + newMessageVector);
      for (Edge<LongWritable, FloatWritable> edge : getEdges()) {
        if (edge.getTargetVertexId().get() != getId().get() && newMessageVector.getNumNondefaultElements() > 0) {
          sendMessage(edge.getTargetVertexId(), newMessage);
        }
      }
    } else {
      voteToHalt();
    }
  }
  public static class RandomWalkVertexInputFormat extends 
    TextVertexInputFormat<LongWritable, VectorWritable, FloatWritable, VectorWritable> {

    @Override
    public VertexReader<LongWritable, VectorWritable, FloatWritable, VectorWritable> createVertexReader(
        InputSplit split, TaskAttemptContext ctx) throws IOException {
      return new RandomWalkVertexReader(textInputFormat.createRecordReader(split, ctx));
    }
  }
  public static class RandomWalkVertexReader extends 
    TextVertexInputFormat.TextVertexReader<LongWritable, VectorWritable, FloatWritable, VectorWritable> {
    
    public static int minNonConceptVertexId = 25425;
    
    public RandomWalkVertexReader(RecordReader<LongWritable, Text> lineRecordReader) {
      super(lineRecordReader);
    }
    public static boolean isConceptVertex(LongWritable vertexId) {
      return vertexId.get() < minNonConceptVertexId; 
    }
    /**
     * define how to read static graph structure from file
     */
    @Override
    public Vertex<LongWritable, VectorWritable, FloatWritable, VectorWritable> getCurrentVertex()
        throws IOException, InterruptedException {
      String line = getRecordReader().getCurrentValue().toString();
      
      Vertex<LongWritable, VectorWritable, FloatWritable, VectorWritable> 
      vertex = BspUtils.createVertex(getContext().getConfiguration());
      
      String[] tokens = line.split("\t");
      LongWritable vertexId = new LongWritable(Long.parseLong(tokens[0]));
      Vector valueVector = new RandomAccessSparseVector(minNonConceptVertexId);
      if (isConceptVertex(vertexId)) {
        valueVector.set((int)vertexId.get(), 1.0);
      }
      VectorWritable vertexValue = new VectorWritable(valueVector);
      
      Map<Long, Float> edgesRaw = Maps.newHashMap();
      Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
      float edgeValueAvg = 0f;
      int edgeNums = 0;
      for (int i = 1; i < tokens.length; i+=2) {
        long targetVertexId = Long.parseLong(tokens[i]);
        float edgeWeight = Float.parseFloat(tokens[i+1]);
        edgesRaw.put(targetVertexId, edgeWeight);
        edgeValueAvg += edgeWeight;
        edgeNums++;
      }
      // normalize
      for (Entry<Long, Float> edge : edgesRaw.entrySet()) {
        edges.put(new LongWritable(edge.getKey()), new FloatWritable(edge.getValue()/edgeValueAvg));
      }
      edgeValueAvg /= (float)edgeNums;
      
      vertex.initialize(vertexId, vertexValue, edges, null);
      return vertex;
    }

    @Override
    public boolean nextVertex() throws IOException, InterruptedException {
      return getRecordReader().nextKeyValue();
    }   
  }
  
  public static class RandomWalkVertexOutputFormat extends 
    TextVertexOutputFormat<LongWritable, VectorWritable, FloatWritable> {

    @Override
    public VertexWriter<LongWritable, VectorWritable, FloatWritable> createVertexWriter(
        TaskAttemptContext ctx) throws IOException, InterruptedException {
      RecordWriter<Text, Text> recordWriter = 
          textOutputFormat.getRecordWriter(ctx);
      return new RandomWalkVertexWriter(recordWriter);
    }
  }
  public static class RandomWalkVertexWriter extends 
    TextVertexWriter<LongWritable, VectorWritable, FloatWritable> {

    public RandomWalkVertexWriter(RecordWriter<Text, Text> lineRecordWriter) {
      super(lineRecordWriter);
    }

    @Override
    public void writeVertex(
        Vertex<LongWritable, VectorWritable, FloatWritable, ?> vertex)
        throws IOException, InterruptedException {
      getRecordWriter().write(
          new Text(vertex.getId().toString()), 
          new Text(vertex.getValue().toString()));
    }
  }
  public static Map<String, String> parseOption(String[] args) {
    Map<String, String> options = Maps.newHashMap();
    for (int i = 0; i < args.length; i++) {
      if (args[i].equals("--input")) {
        options.put(args[i], args[i+1]);
      } else if (args[i].equals("--output")) {
        options.put(args[i], args[i+1]);
      } else if (args[i].equals("--w")) {
        options.put(args[i], args[i+1]);
      } else if (args[i].equals("--gamma")) {
        options.put(args[i], args[i+1]);
      } else if (args[i].equals("--iteration")) {
        options.put(args[i], args[i+1]);
      } else if (args[i].equals("--minNonConceptVertexId")) {
        options.put(args[i], args[i+1]);
      } else if (args[i].equals("--zklist")) {
        options.put(args[i], args[i+1]);
      }
    }
    for (Entry<String, String> option : options.entrySet()) {
      System.out.println(option.getKey() + "\t" + option.getValue());
    }
    return options;
  }
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RandomWalkVertex(), args);
  }
  @Override
  public int run(String[] args) throws Exception {
    Map<String, String> options = parseOption(args);
    RandomWalkVertexReader.minNonConceptVertexId = Integer.parseInt(options.get("--minNonConceptVertexId"));
    
    GiraphJob job = new GiraphJob(getConf(), getClass().getName());
    job.getInternalJob().setJarByClass(RandomWalkVertex.class);
    job.setVertexClass(getClass());
    job.setVertexInputFormatClass(RandomWalkVertexInputFormat.class);
    job.setVertexOutputFormatClass(RandomWalkVertexOutputFormat.class);
    FileInputFormat.addInputPath(job.getInternalJob(), new Path(options.get("--input")));
    FileOutputFormat.setOutputPath(job.getInternalJob(), new Path(options.get("--output")));
    job.getConfiguration().setFloat(GAMMA, Float.parseFloat(options.get("--gamma")));
    job.getConfiguration().setInt(ITERATION, Integer.parseInt(options.get("--iteration")));
    job.getConfiguration().getInt(MIN_NON_CONCEPT_VERTEX_ID, Integer.parseInt(options.get("--minNonConceptVertexId")));
    job.setWorkerConfiguration(1, Integer.parseInt(options.get("--w")), 100.0f);
    job.setZooKeeperConfiguration(options.get("--zklist") == null ? "20.20.20.31" : options.get("--zklist"));
    if (job.run(true) == true) {
      return 0;
    } else {
      return -1;
    }
  }
}
