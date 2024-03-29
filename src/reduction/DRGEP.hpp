/*-----------------------------------------------------------------------*\
|    ___                   ____  __  __  ___  _  _______                  |
|   / _ \ _ __   ___ _ __ / ___||  \/  |/ _ \| |/ / ____| _     _         |
|  | | | | '_ \ / _ \ '_ \\___ \| |\/| | | | | ' /|  _| _| |_ _| |_       |
|  | |_| | |_) |  __/ | | |___) | |  | | |_| | . \| |__|_   _|_   _|      |
|   \___/| .__/ \___|_| |_|____/|_|  |_|\___/|_|\_\_____||_|   |_|        |
|        |_|                                                              |
|                                                                         |
|   Author: Alberto Cuoci <alberto.cuoci@polimi.it>                       |
|   CRECK Modeling Group <http://creckmodeling.chem.polimi.it>            |
|   Department of Chemistry, Materials and Chemical Engineering           |
|   Politecnico di Milano                                                 |
|   P.zza Leonardo da Vinci 32, 20133 Milano                              |
|                                                                         |
|-------------------------------------------------------------------------|
|                                                                         |
|   This file is part of OpenSMOKE++ framework.                           |
|                                                                         |
|	License                                                               |
|                                                                         |
|   Copyright(C) 2018  Alberto Cuoci                                      |
|   OpenSMOKE++ is free software: you can redistribute it and/or modify   |
|   it under the terms of the GNU General Public License as published by  |
|   the Free Software Foundation, either version 3 of the License, or     |
|   (at your option) any later version.                                   |
|                                                                         |
|   OpenSMOKE++ is distributed in the hope that it will be useful,        |
|   but WITHOUT ANY WARRANTY; without even the implied warranty of        |
|   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         |
|   GNU General Public License for more details.                          |
|                                                                         |
|   You should have received a copy of the GNU General Public License     |
|   along with OpenSMOKE++. If not, see <http://www.gnu.org/licenses/>.   |
|                                                                         |
\*-----------------------------------------------------------------------*/

#include <queue>

namespace OpenSMOKE
{
	DRGEP::DRGEP(	OpenSMOKE::ThermodynamicsMap_CHEMKIN* thermodynamicsMapXML,
					OpenSMOKE::KineticsMap_CHEMKIN* kineticsMapXML) :

		thermodynamicsMapXML_(*thermodynamicsMapXML),
		kineticsMapXML_(*kineticsMapXML),
		kinetics_graph_(*(new KineticsGraph(&thermodynamicsMapXML_, &kineticsMapXML_)))
	{
		epsilon_ = 1.e-2;
		T_threshold_ = 310.;
	
		NS_ = thermodynamicsMapXML_.NumberOfSpecies();
		NR_ = kineticsMapXML_.NumberOfReactions();

		important_species_.resize(NS_);
		important_reactions_.resize(NR_);

		number_important_species_ = NS_;
		number_unimportant_reactions_ = 0;

		// Build a full matrix of net stoichiometric coefficients nu = nuB - nuF
		Eigen::MatrixXd nu(NR_, NS_);
		{
			// Be careful: eigen vectors and matrices are 0-index based
			// Be careful: if the kinetic scheme is large, this matrix, since it is full, can be very memory expensive
			//             Example: 10^3 species, 10^4 reactions = size of the matrix 10^7 elements!
			//             This is the reason why we store stoichiometric matrices in sparse format.
			//             Of course the advantage of having a full matrix, is that you access the elements directly, without
			//             using iterators and pointers, as reported above
			nu.setZero();

			// Loop over all the reactions (product side)
			for (int k = 0; k < kineticsMapXML_.stoichiometry().stoichiometric_matrix_products().outerSize(); ++k)
			{
				// Loop over all the non-zero stoichiometric coefficients (product side) of reaction k
				for (Eigen::SparseMatrix<double>::InnerIterator it(kineticsMapXML_.stoichiometry().stoichiometric_matrix_products(), k); it; ++it)
				{
					nu(it.row(), it.col()) += it.value();
				}
			}

			// Loop over all the reactions (product side)
			for (int k = 0; k < kineticsMapXML_.stoichiometry().stoichiometric_matrix_reactants().outerSize(); ++k)
			{
				// Loop over all the non-zero stoichiometric coefficients (product side) of reaction k
				for (Eigen::SparseMatrix<double>::InnerIterator it(kineticsMapXML_.stoichiometry().stoichiometric_matrix_reactants(), k); it; ++it)
				{
					nu(it.row(), it.col()) -= it.value();
				}
			}
		}

		// Build the delta matrix (dense matrix) used by the DRGEP method
		Eigen::MatrixXd delta(NR_, NS_);
		{
			for (unsigned int i = 0; i < NR_; i++)
			{
				for (unsigned int j = 0; j < NS_; j++)
					delta(i, j) = (nu(i, j) == 0) ? 0 : 1;
			}
		}

		// Sparse delta matrix, 1 means the species is involved in the reaction, (NR x NS)
		delta_sparse_.resize(NS_, NR_);
		{
			typedef Eigen::Triplet<double> T;
			std::vector<T> tripletList;
			tripletList.reserve(NR_ * 4);
			for (unsigned int i = 0; i < NR_; i++)
				for (unsigned int j = 0; j < NS_; j++)
				{
					if (delta(i, j) != 0.)
						tripletList.push_back(T(j, i, 1.));
				}

			delta_sparse_.setFromTriplets(tripletList.begin(), tripletList.end());
		}
	}

	void DRGEP::PrepareKineticGraph(const std::vector<std::string> names_key_species)
	{
		iScaling_ = true;

		scaling_factor_.resize(index_key_species_.size());
		if (iScaling_ == true) scaling_factor_.reserve(5000);

		target_oic_.reserve(5000);
		kinetics_graph_.SetKeySpecies(names_key_species);

		local_dic_.resize(thermodynamicsMapXML_.NumberOfSpecies(), thermodynamicsMapXML_.NumberOfSpecies());
		local_dic_.setZero();
	}

	void DRGEP::ResetKineticGraph()
	{
		important_species_.resize(NS_);
		important_reactions_.resize(NR_);
		number_important_species_ = NS_;
		number_unimportant_reactions_ = 0;

		scaling_factor_.resize(index_key_species_.size());
		if (iScaling_ == true) scaling_factor_.reserve(5000);

		target_oic_.resize(0);
		target_oic_.reserve(5000);
	}

	void DRGEP::SetKeySpecies(const std::vector<std::string> names_key_species)
	{
		names_key_species_ = names_key_species;
		index_key_species_.resize(names_key_species.size());
		for (unsigned int i = 0; i < names_key_species.size(); i++)
			index_key_species_[i] = thermodynamicsMapXML_.IndexOfSpecies(names_key_species[i]) - 1;
	}

	void DRGEP::SetKeySpecies(const std::vector<unsigned int> key_species)
	{
		index_key_species_ = key_species;
		names_key_species_.resize(index_key_species_.size());
		for (unsigned int i = 0; i < names_key_species_.size(); i++)
			names_key_species_[i] = thermodynamicsMapXML_.NamesOfSpecies()[index_key_species_[i]];
	}

	void DRGEP::SetEpsilon(const double epsilon)
	{
		epsilon_ = epsilon;
	}

	void DRGEP::SetTemperatureThreshold(const double T_threshold)
	{
		T_threshold_ = T_threshold;
	}

	void DRGEP::Analysis(const double T, const double P_Pa, const OpenSMOKE::OpenSMOKEVectorDouble& c)
	{
		ResetKineticGraph();

		// Calculate
		{
			OpenSMOKE::OpenSMOKEVectorDouble rnet(kineticsMapXML_.NumberOfReactions());

			// Now we know T, P and composition.
			// We have to pass those data to the thermodynamic and kinetic maps
			thermodynamicsMapXML_.SetTemperature(T);
			thermodynamicsMapXML_.SetPressure(P_Pa);
			kineticsMapXML_.SetTemperature(T);
			kineticsMapXML_.SetPressure(P_Pa);

			kineticsMapXML_.ReactionRates(c.GetHandle());
			kineticsMapXML_.GiveMeReactionRates(rnet.GetHandle());	// [kmol/m3/s]

			OpenSMOKE::OpenSMOKEVectorDouble P(thermodynamicsMapXML_.NumberOfSpecies());
			OpenSMOKE::OpenSMOKEVectorDouble D(thermodynamicsMapXML_.NumberOfSpecies());
			kineticsMapXML_.ProductionAndDestructionRates(P.GetHandle(), D.GetHandle());

			// Exclude cold cells
			if (T < T_threshold_)
			{
				P = 0.;
				D = 0.;
				rnet = 0.;
			}

			// -------------------------------------------------------------------------
			// Pepiot-Desjardins, P., & Pitsch, H. (2008). Comb Flame, 154(1-2), 67-81
			// -------------------------------------------------------------------------
			local_dic_.setZero();
			for (int j = 0; j < kineticsMapXML_.NumberOfReactions(); j++)
				for (Eigen::SparseMatrix<double>::InnerIterator it_nu(kinetics_graph_.stoichiometric_matrix_overall(), j); it_nu; ++it_nu)
					for (Eigen::SparseMatrix<double>::InnerIterator it_bool(kinetics_graph_.stoichiometric_matrix_overall(), j); it_bool; ++it_bool)
						local_dic_(it_nu.row(),it_bool.row()) += OpenSMOKE::Abs(rnet[j + 1] * it_nu.value());

			for (int i = 0; i < thermodynamicsMapXML_.NumberOfSpecies(); i++)
			{
				if (OpenSMOKE::Max(P[i + 1], D[i + 1]) > 1.e-20)
					for (int j = 0; j < thermodynamicsMapXML_.NumberOfSpecies(); j++)
						local_dic_(i,j) /= (P[i + 1] + D[i + 1]); // Lu & Law use (P+D), Pepiot & Pitsch use Max(P,D)
				else
					for (int j = 0; j < thermodynamicsMapXML_.NumberOfSpecies(); j++)
						local_dic_(i,j) = 0.;
			}

			for (int i = 0; i < thermodynamicsMapXML_.NumberOfSpecies(); i++)
				local_dic_(i,i) = 1.;

			if (iScaling_ == true)
			{
				//Calculating the scaling factor
				for (int i = 0; i < scaling_factor_.size(); i++)
				{
					int index = thermodynamicsMapXML_.IndexOfSpecies(names_key_species_[i]) - 1;

					std::vector<double> atomic_scaling(thermodynamicsMapXML_.elements().size());
					std::vector<double> atom_flux(thermodynamicsMapXML_.elements().size());
					std::fill(atomic_scaling.begin(), atomic_scaling.end(), 0.);
					std::fill(atom_flux.begin(), atom_flux.end(), 0.);

					//Calculating P_a_t
					for (int k = 0; k < thermodynamicsMapXML_.elements().size(); k++)
					{
						for (int j = 0; j < thermodynamicsMapXML_.NumberOfSpecies(); j++)
							atom_flux[k] += thermodynamicsMapXML_.atomic_composition()(j, k)*OpenSMOKE::Max(0., P[j + 1] - D[j + 1]);
					}

					//Calculating alfa_a_t
					for (int k = 0; k < thermodynamicsMapXML_.elements().size(); k++)
					{
						if (atom_flux[k] > 0)
							atomic_scaling[k] = (thermodynamicsMapXML_.atomic_composition()(index, k) *
								OpenSMOKE::Abs(P[index + 1] - D[index + 1])) / atom_flux[k];
					}

					//Calculating scaling factor
					double alfa_t = 0.;
					for (int k = 0; k < thermodynamicsMapXML_.elements().size(); k++)
						alfa_t = OpenSMOKE::Max(alfa_t, atomic_scaling[k]);
					scaling_factor_[i].push_back(alfa_t);

					// Clean
					atomic_scaling.clear();
					atom_flux.clear();
				}
			}
		}

		kinetics_graph_.SetWeights(local_dic_);
		target_oic_.push_back(kinetics_graph_.ShortestPaths());

		{
			oic_values_.resize(index_key_species_.size());
			for (int i = 0; i < index_key_species_.size(); i++)
				ChangeDimensions(thermodynamicsMapXML_.NumberOfSpecies(), &oic_values_[i], true);

			//Normalize scaling factors, if enabled, and apply to oic
			if (iScaling_ == true)
			{
				for (int i = 0; i < index_key_species_.size(); i++)
				{
					double max_alfa_t = *std::max_element(scaling_factor_[i].begin(), scaling_factor_[i].end());
					for (int k = 0; k < scaling_factor_[i].size(); k++)
						scaling_factor_[i][k] /= max_alfa_t;
				}

				for (int k = 0; k < target_oic_.size(); k++)
					for (int i = 0; i < index_key_species_.size(); i++)
						for (int j = 0; j < thermodynamicsMapXML_.NumberOfSpecies(); j++)
							target_oic_[k][i][j] *= scaling_factor_[i][k];
			}

			for (int k = 0; k < target_oic_.size(); k++)
				for (int i = 0; i < index_key_species_.size(); i++)
					for (int j = 0; j < thermodynamicsMapXML_.NumberOfSpecies(); j++)
						oic_values_[i][j+1] = OpenSMOKE::Max(oic_values_[i][j + 1], target_oic_[k][i][j]);
		}

		std::vector<OpenSMOKE::OpenSMOKEVectorDouble> flux_ranking = oic_values_;

		// Reset important species and important reactions
		important_species_.assign(NS_, false);
		important_reactions_.assign(NR_, true);

		// Important species: key species
		for (int j = 0; j < index_key_species_.size(); j++)
			important_species_[index_key_species_[j]] = true;

		// Important species: fluxes
		for (int j = 0; j < flux_ranking.size(); j++)
			for (int k = 0; k < thermodynamicsMapXML_.NumberOfSpecies(); k++)
				if (flux_ranking[j][k+1] > epsilon_ ) // [j]
					important_species_[k] = true;

		// Important reactions
		for (int k = 0; k<delta_sparse_.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(delta_sparse_, k); it; ++it)
			{
				if (important_species_[it.row()] == false)
					important_reactions_[k] = false;
			}
		}

		// Count important species and reactions
		number_important_species_ = std::count(important_species_.begin(), important_species_.end(), true);
		number_unimportant_reactions_ = std::count(important_reactions_.begin(), important_reactions_.end(), false);

		// Vector containing the indices of important species (zero-based)
		{
			indices_important_species_.resize(number_important_species_);
			unsigned int count = 0;
			for (unsigned int k = 0; k < NS_; k++)
				if (important_species_[k] == true)
				{
					indices_important_species_[count] = k;
					count++;
				}
		}

		// Vector containing the indices of unimportant species (zero-based)
		{
			indices_unimportant_reactions_.resize(number_unimportant_reactions_);
			unsigned int count = 0;
			for (unsigned int k = 0; k < NR_; k++)
				if (important_reactions_[k] == false)
				{
					indices_unimportant_reactions_[count] = k;
					count++;
				}
		}
	}
}