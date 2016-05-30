wiki_input = ['enwiki-20160501-pages-articles1', 'enwiki-20160501-pages-articles10', 'enwiki-20160501-pages-articles20']
output = File.open("output", 'w')

wiki_input.each {|file_name|
  article = ""
  #i = 0
  File.new(file_name).each {|line|
   #break if(i == 10000) 
   if(line =~ /<doc[^>]*>/) 
     article =""
   elsif(line =~ /<\/doc>/)
     output.write article+"\n"
     # puts article
   else
     article += line.chomp + " "
   end
   #i+=1
  }
}
output.close
