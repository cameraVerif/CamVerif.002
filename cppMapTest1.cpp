#include <map>
#include <set>
#include <tuple>

#include <iostream>
#include <fstream>

int main() {
  // Initialize an empty map
  std::map<std::string, std::set<std::tuple<int, int,
                                            int, int,
                                            int, int>>> hashmap;

  // Add elements to the hashmap
  hashmap["key1"] = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {117, 118, 119, 120, 121, 122}};
  hashmap["key2"] = {{13, 14, 15, 16, 17, 18}, {19, 20, 21, 22, 23, 24}};

  // Access elements for a given key
  auto elements = hashmap["key1"];
    std::cout << elements.size()<< std::endl;


// Print the elements of the hashmap
  for (const auto &keyValuePair : hashmap) {
    const std::string &key = keyValuePair.first;
    const std::set<std::tuple<int, int,int, int,int, int>> &sets = keyValuePair.second;
    std::cout << "Key: " << key << std::endl;
    std::set<std::tuple<int, int,int, int,int, int>>:: iterator it;
     for( it = sets.begin(); it!=sets.end(); ++it){
            // int ans = *it;
            // cout << ans << endl;
            std::tuple<int, int,int, int,int, int> triple = *it;
            int a, b, c, d, e, f;
            std::tie(a, b, c, d, e, f) = triple;
            std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << std::endl;

        }
  }

    std::cout<<"\n\nInserting ..........\n\n";
  // Insert an element into the map
  std::tuple<int, int, int, int, int, int> element = {81, 82, 83, 84, 85, 86};
  hashmap["key1"].insert(element);

  hashmap["key44"].insert(element);



// Print the elements of the hashmap
  for (const auto &keyValuePair : hashmap) {
    const std::string &key = keyValuePair.first;
    const std::set<std::tuple<int, int,int, int,int, int>> &sets = keyValuePair.second;
    std::cout << "Key: " << key << std::endl;
    std::set<std::tuple<int, int,int, int,int, int>>:: iterator it;
     for( it = sets.begin(); it!=sets.end(); ++it){
            // int ans = *it;
            // cout << ans << endl;
            std::tuple<int, int,int, int,int, int> triple = *it;
            int a, b, c, d, e, f;
            std::tie(a, b, c, d, e, f) = triple;
            std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << std::endl;

        }
  }




  // Check if a key exists in the hashmap
  if (hashmap.count("key3") > 0) {
    std::cout << "key3 exists" << std::endl;
  } else {
    std::cout << "key3 does not exist" << std::endl;
  }



 std::ofstream outfile("example.txt",std::ios::out | std::ios::trunc);
  if (outfile.is_open()) {

    for (const auto &keyValuePair : hashmap) {
        const std::string &key = keyValuePair.first;
        const std::set<std::tuple<int, int,int, int,int, int>> &sets = keyValuePair.second;
        std::cout << "Key: " << key << std::endl;
        std::cout<< "Number of elements : "<<sets.size() << std::endl;
        outfile << key << std::endl;
        outfile << sets.size() << std::endl;
        std::set<std::tuple<int, int,int, int,int, int>>:: iterator it;
        for( it = sets.begin(); it!=sets.end(); ++it){
                // int ans = *it;
                // cout << ans << endl;
                std::tuple<int, int,int, int,int, int> triple = *it;
                int a, b, c, d, e, f;
                std::tie(a, b, c, d, e, f) = triple;
                outfile << a << "," << b << "," << c << "," << d << "," << e << "," << f << std::endl;

            }
    }


    outfile << "This is some text that will be written to a file." << std::endl;
    outfile << "Here's another line of text." << std::endl;
    outfile.close();
  } else {
    std::cout << "Unable to open file." << std::endl;
  }

  return 0;
}
