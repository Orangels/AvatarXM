//
// Created by Orangels on 2023/9/6.
//

#ifndef AVATARXM_UTILS_H
#define AVATARXM_UTILS_H
#include <iostream>
#include <fstream>
#include <vector>


using namespace std;

namespace avatarUtiles{
    inline std::vector<std::string> readClassNames(string label)
    {
        std::string labels_txt_file = label;
        std::vector<std::string> classNames;

        std::ifstream fp(labels_txt_file);
        if (!fp.is_open())
        {
            printf("could not open file...\n");
            exit(-1);
        }
        std::string name;
        while (!fp.eof())
        {
            std::getline(fp, name);
            if (name.length())
                classNames.push_back(name);
        }
        fp.close();
        return classNames;
    }
}

#endif //AVATARXM_UTILS_H
