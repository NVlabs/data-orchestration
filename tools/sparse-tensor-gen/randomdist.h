
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <assert.h>
#include <algorithm>
#include <random>
#include <math.h>

#define VAL_FACTOR 8
#define VAL_SPREAD 8
#define VAL_MAX 255  


class DegreeInfo
{
public:
    int id_;
    int degree_;
public:
    DegreeInfo();
    DegreeInfo(int id, int degree = 0) : id_(id), degree_(degree) {}
    void Increment() { degree_++; }
    
    bool operator < (const DegreeInfo& cmp) const
    {
        if (degree_ > cmp.degree_){
            return true;
        }

        return false;
        
    }
};


class Node
{
public:
    int x_;
    int y_;
    int val_;
public:
    Node();
    Node(int x, int y, int val) : x_(x), y_(y), val_(val) {}
    bool operator < (const Node& node) const
    {
        if (x_ < node.x_){
            return true;
        }
        else if (x_ == node.x_ && y_ < node.y_)
        {
            return true;
        }
        else {
            return false;
        }
    }
};



class CTNonzeroGenerator
{
public:
    int dim_x_;
    int dim_y_;
    std::vector<Node> nonzeroes_;

    std::default_random_engine generator;
    std::binomial_distribution<int> bdistribution_;
    std::normal_distribution<double> ndistribution_;

    int option_symmetric_; //0 == false
    int option_distribution_;

public:
    CTNonzeroGenerator(){}

    CTNonzeroGenerator(int dim_x, int dim_y) : dim_x_(dim_x), dim_y_(dim_y), option_symmetric_(false), option_distribution_(0)  {
        bdistribution_ = std::binomial_distribution<int> (dim_y,0.5);
        ndistribution_ = std::normal_distribution<double> (dim_y/2,dim_y/8);
    }

    void SetDistribution(int mode)
    {
        option_distribution_ = mode;
    }

    void SetSymmetric()
    {
        // do check to make sure that it is safe to do symmetric matrix (if requested)
        assert (dim_x_ == dim_y_ && "Symmetric requested but dim sizes aren't equal!");
        option_symmetric_ = true;
    }

    void AddNonzero(int x, int y, int val)
    {
        Node new_node = Node(x, y, val);
        nonzeroes_.push_back(new_node);
    }

    int GetNonzeroes()
    {
        return nonzeroes_.size();
    }

    bool CheckBucketRange(int threshold, int maximum)
    {
        int roll = rand() % maximum;
        if (roll < threshold){
            return true;
        }
        return false;
    }


    //frac_nz is 1/X nz chance
    int RandomUniform(int max)
    {
        int value = (rand() % max) + 1;
        printf("random frac from %d = %d \n",max, value);
        return value;
    }

    //float       pow ( float base, float exp );
    //Where: y = uniform variate, n = distribution power, x0 and x1 = range of the distribution, x = power-law distributed variate.
    float RandomPowerLaw(float x0, float x1, int n)
    {
        int maximum = 1024*1024*1024;
        int roll = rand() % maximum;
        double y = double(roll) / double(maximum);
        double final_base = (pow(x1,(n+1)) - pow(x0,(n+1)))*y + pow(x0,(n+1));
        double final_pow = (1/(double(n)+1));

        double x = pow(final_base,final_pow);
        //printf("roll: %2.2f  base: %2.2f  power: %2.2f final: %2.2f\n", y, final_base, final_pow, x);
        return x;
    }

    //implement a exponential distribution e.g.  1/2 are 0, 1/4 are 1, 1/8 are 2 (ala taylor series) per nonzero buckets
    //Where: frac is 1/frac nonzero for this stage
    // iterate and roll until we fail (failure is rolling something nonzero). 
    int RandomExponential(int size, float frac)
    {
        int maximum = 1000;
        float threshold = frac*maximum;
        int count = 0;
        int roll = rand() % maximum;
        while(roll < int(threshold) && count < size && int(threshold) < maximum)
        {
            roll = rand() % maximum;
            count++;
            //printf(" thres: %2.2f (%d / %d) scale: %2.2f count: %d\n", threshold/maximum,  int(threshold), maximum, scale, count);
        }
        //printf("get frac: %2.2f scale: %2.2f count: %d\n", frac, scale, count);
        return count;
    }

    //implement a longtail exponential distribution e.g.  1/2 are 0, 1/4 are 1, 1/8 are 2 (ala taylor series) per nonzero buckets
    //Where: size is max_count, frac_min is how much passes each stage (1/frac extra nonzero for this stage)
    //  frac_max is the saturation of 
    //  scale is how much is reduced each time. 
    // iterate and roll until we fail (failure is rolling something nonzero). 
    int RandomLongTail(int size, float frac_min, float frac_max, float scale)
    {
        int maximum_roll = 1000;
        float curr_threshold = frac_min*maximum_roll;
        float max_threshold = frac_max*maximum_roll;
        int count = 0;
        int roll = rand() % maximum_roll;
    //printf(" thres: %2.2f (%d / %d) scale: %2.2f count: %d\n", threshold/maximum,  int(threshold), maximum, scale, count);
        while(roll < int(curr_threshold) && count < size)
        {
            if (curr_threshold < max_threshold)
            {
                curr_threshold += float(maximum_roll - curr_threshold) * scale;
            }
            roll = rand() % maximum_roll;
            count++;
            //printf(" thres: %2.2f (%d / %d) scale: %2.2f count: %d\n", curr_threshold/maximum_roll,  int(curr_threshold), maximum_roll, scale, count);
        }
        //printf("get frac: %2.2f scale: %2.2f count: %d\n", frac, scale, count);
        return count;
    }


    int GetDistributionRandom()
    {
        int bucket;
        switch(option_distribution_)
        {
            //uniform
            default:
            case 0:
                bucket = RandomUniform(dim_y_);            
                return bucket;            
            //binomial
            case 1:
                bucket = bdistribution_(generator);
                return bucket;
            //normal
            case 2:
            {
                double number = ndistribution_(generator);
                if (number<=0.0){
                    number = 0.0;
                } 
                else if (number>dim_y_)
                {
                    number = dim_y_;
                }
                bucket = int(number);
                return bucket;
            }
            //power
            case 3:
            {
                float pval = RandomPowerLaw(0, dim_y_, 20);
                bucket = dim_y_-pval;            
                return bucket;
            }
            //exponential
            case 4:
                return bucket;
            //longtail (scaled exponential)
            case 5:
                bucket = RandomLongTail(dim_y_, 0.2, 0.9, 0.3);
                return bucket;        
        }
    }

    void Generate()
    {
        

        for (int i = 0; i < dim_x_; i++)
        {

            //handle symmetric case
            int j_start = 0;
            if (option_symmetric_ == true)
            {
                j_start = i;
            } 

            int bucket = GetDistributionRandom();
            for (int j = j_start; j < dim_y_; j++)
            {
                if (true == CheckBucketRange(bucket, dim_y_))
                {
                    AddNonzero(i,j,1);
                    //generate symmetry for non diagonals
                    if (option_symmetric_ == true && i != j)
                    {
                        AddNonzero(j,i,1);
                    }
                }
            }
        }

        if (option_symmetric_ == true)
        {
            Sort();
        } 

    }

    void Sort()
    {
        std::sort(nonzeroes_.begin(),nonzeroes_.end());
    }

    void SortDegree()
    {
        Sort();

        printf("** DegreeSort **\n");
        std::vector<DegreeInfo> rows;
        printf("** - Init **\n");
        for (int i = 0; i < dim_x_; i++)
        {
            rows.push_back(DegreeInfo(i));
        }

        printf("** - Degree Count **\n");
        for (Node node: nonzeroes_)
        {
            rows[node.x_].Increment();
        }

        printf("** - Sort Histo **\n");
        std::sort(rows.begin(),rows.end());

        printf("** - Build Translation **\n");
        std::vector<int> translation;
        translation.resize(dim_x_);
        for (int i = 0; i < rows.size(); i++)
        {
            //printf("   - row %d  count:%d\n", rows[i].id_, rows[i].degree_);        
            translation[rows[i].id_] = i;
        }
        

        //replace
        //
        printf("** - Translating **\n");        
        for (Node& node: nonzeroes_)
        {
            //translate x
            node.x_ = translation[node.x_];
            if (option_symmetric_ == true)
            {
                node.y_ = translation[node.y_];
            }
        }
        
        printf("** - ReSort COO **\n");        
        Sort();
    }


    void Dump()
    {
        printf("** Dumping Nonzeroes **\n");
        for (Node node: nonzeroes_)
        {
            printf("  (%d, %d) - %d\n", node.x_, node.y_, node.val_);        
        }
    }

    void Dump2D ()
    {
        printf("** Dumping Nonzeroes **\n");

        int count = 0;
        int index = 0;
        for (int i = 0; i < dim_x_; i++)
        {
            for (int j = 0; j < dim_y_; j++)
            {
                if (index < nonzeroes_.size() && nonzeroes_[index].x_ == i && nonzeroes_[index].y_ == j)
                {
                    printf(" %d", nonzeroes_[index].val_);
                    count++;
                    index++;
                }
                else {
                    printf(" 0");                
                }
            }

            printf(": (count:%d) \n", count);

            count = 0;
        }


    }

    void Histo()
    {
        std::vector<int> row_count;
        std::vector<int> col_count;
        row_count.resize(dim_x_);
        col_count.resize(dim_y_);
        printf("** Generating histogram **\n");
        for (Node node: nonzeroes_)
        {
            row_count[node.x_]++;
            col_count[node.y_]++;
        }

        std::vector<int> row_histo;
        std::vector<int> col_histo;
        row_histo.resize(dim_y_);
        col_histo.resize(dim_x_);
        for (int count: row_count)
        {
            row_histo[count]++;
        }

        for (int count: col_count)
        {
            col_histo[count]++;
        }

        //print histos
        printf("Row dimension (across rows):\n");
        printf("bucket(count),  ");
        int index = 0;
        for (int count: row_histo)
        {
            if (count != 0){
                printf("%d(%d),  ", index, count);
            }
            index++;
        }
        printf("\n");
        printf("Col dimension (across cols):\n");
        printf("bucket(count),  ");
        index = 0;
        for (int count: col_histo)
        {
            if (count != 0){
                printf("%d(%d),  ", index, count);
            }
            index++;
        }
        printf("\n");

    }

};



