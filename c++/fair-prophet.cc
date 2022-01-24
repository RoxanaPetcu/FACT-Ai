// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fair-prophet.h"

namespace fair_secretary {

using std::vector;

SecretaryInstance FairProphetAlgorithm::ComputeSolution(
    const vector<SecretaryInstance>& elements,
    std::vector<UniformDistribution>& distributions,
    const vector<double>& q) {
  double sum = 0;
  for (int i = 0; i < elements.size(); i++) {
    if (elements[i].value >= distributions[elements[i].type].Reverse(
                                 1.0 - (q[i] / (2 - sum)))) {
      return elements[i];
    }
    sum += q[i];
  }
  return SecretaryInstance(-1, -1);
}

SecretaryInstance FairProphetAlgorithm::ComputeSolutionIID(
    const vector<SecretaryInstance>& elements,
    std::vector<UniformDistribution>& distributions,
    const vector<double>& q) {
  double p = (2.0 / 3.0) / elements.size();
  for (int i = 0; i < elements.size(); i++) {
    if (elements[i].value >=
        distributions[elements[i].type].Reverse(1.0 - p / (1 - p * i))) {
      return elements[i];
    }
  }
  return SecretaryInstance(-1, -1);
}

}  // namespace fair_secretary
