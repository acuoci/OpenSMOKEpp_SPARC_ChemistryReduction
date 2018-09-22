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

// Include OpenMP Header file
#if defined(_OPENMP)
#include <omp.h>
#endif

// Neural networks from MATLAB(R)
//#include "neural/myNeuralNetworkBasedOnPCA.h"

// OpenSMOKE++ Definitions
#include "OpenSMOKEpp"

// Thermodynamics
#include "kernel/thermo/Species.h"
#include "kernel/thermo/ThermoPolicy_CHEMKIN.h"
#include "kernel/thermo/ThermoReader.h"
#include "kernel/thermo/ThermoReaderPolicy_CHEMKIN.h"

// Kinetics
#include "kernel/kinetics/ReactionPolicy_CHEMKIN.h"

// Preprocessing
#include "preprocessing/PreProcessorSpecies.h"
#include "preprocessing/PreProcessorKinetics.h"
#include "preprocessing/PreProcessorKineticsPolicy_CHEMKIN.h"
#include "preprocessing/PreProcessorSpeciesPolicy_CHEMKIN_WithoutTransport.h"
#include "preprocessing/PreProcessorSpeciesPolicy_CHEMKIN_WithTransport.h"

// CHEMKIN maps
#include "maps/Maps_CHEMKIN"
#include "reduction/DRG.h"
#include "reduction/DRGEP.h"

// Grammar
#include "reduction/Grammar_StaticReduction.h"

void PrintMatrixOfPresence(	const boost::filesystem::path& path_output_folder,
							const unsigned int ndata,
							const Eigen::VectorXi& number_important_species,
							const Eigen::MatrixXi& important_species);

void PrintErrorAnalysis(const boost::filesystem::path& path_output_folder,
						const unsigned int ndata, const unsigned int nclusters,
						const Eigen::VectorXi& number_important_species,
						const Eigen::MatrixXi& important_species,
						const std::vector< std::vector<int> > belonging,
						const std::vector<std::string> species_names,
						const double retained_threshold,
						Eigen::MatrixXi& retained_species);

void WriteKineticMechanisms(const OpenSMOKE::ThermodynamicsMap_CHEMKIN& thermodynamicsMap,
							const OpenSMOKE::KineticsMap_CHEMKIN& kineticsMap,
							const boost::filesystem::path& path_output_folder,
							const boost::filesystem::path& chemkin_thermodynamic_file,
							const boost::filesystem::path& chemkin_kinetics_file,
							const Eigen::MatrixXi& retained_species,
							const Eigen::MatrixXi& retained_reactions);

void SelectImportantReactions(	OpenSMOKE::KineticsMap_CHEMKIN& kineticsMap,
								Eigen::MatrixXi& retained_species,
								Eigen::MatrixXi& retained_reactions);

void PreprocessKineticMechanisms(	const unsigned int nclusters,
									const boost::filesystem::path& path_output_folder,
									const boost::filesystem::path& chemkin_thermodynamic_file,
									const boost::filesystem::path& chemkin_transport_file);
/*
double EpsilonDRG(const double T)
{
	const double w = 300.;
	const double epsMax = 0.05;
	const double epsMin = 0.01;
	const double Tm = 1000.;

	return epsMax - (epsMax - epsMin)*0.50*(1. + std::tanh((T-Tm)/w));
}

double EpsilonDRGEP(const double T)
{
	const double w = 300.;
	const double epsMax = 0.020;
	const double epsMin = 0.005;
	const double Tm = 1000.;

	return epsMax - (epsMax - epsMin)*0.50*(1. + std::tanh((T - Tm) / w));
}
*/
int main(int argc, char** argv)
{
	boost::filesystem::path executable_file = OpenSMOKE::GetExecutableFileName(argv);
	boost::filesystem::path executable_folder = executable_file.parent_path();

	OpenSMOKE::OpenSMOKE_logo("OpenSMOKEpp_SPARC_ChemistryReduction", "Alberto Cuoci (alberto.cuoci@polimi.it)");

	//unsigned int max_number_allowed_species = 100000;
	//OpenSMOKE::OpenSMOKE_CheckLicense(executable_folder, "OpenSMOKE_StaticReduction", max_number_allowed_species);

	std::string input_file_name_ = "input.dic";
	std::string main_dictionary_name_ = "StaticReduction";
	unsigned int number_threads = 1;

	// Program options from command line
	{
		namespace po = boost::program_options;
		po::options_description description("Options for the OpenSMOKEpp_SPARC_ChemistryReduction");
		description.add_options()
			("help", "print help messages")
			("np", po::value<unsigned int>(), "number of threads (default 1")
			("input", po::value<std::string>(), "name of the file containing the main dictionary (default \"input.dic\")")
			("dictionary", po::value<std::string>(), "name of the main dictionary to be used (default \"StaticReduction\")");

		po::variables_map vm;
		try
		{
			po::store(po::parse_command_line(argc, argv, description), vm); // can throw 

			if (vm.count("help"))
			{
				std::cout << "Basic Command Line Parameters" << std::endl;
				std::cout << description << std::endl;
				return OPENSMOKE_SUCCESSFULL_EXIT;
			}

			if (vm.count("np"))
				number_threads = vm["np"].as<unsigned int>();

			if (vm.count("input"))
				input_file_name_ = vm["input"].as<std::string>();

			if (vm.count("dictionary"))
				main_dictionary_name_ = vm["dictionary"].as<std::string>();

			po::notify(vm); // throws on error, so do after help in case  there are any problems 
		}
		catch (po::error& e)
		{
			std::cerr << "Fatal error: " << e.what() << std::endl << std::endl;
			std::cerr << description << std::endl;
			return OPENSMOKE_FATAL_ERROR_EXIT;
		}
	}

	// Defines the grammar rules
	OpenSMOKE::Grammar_StaticReduction grammar_staticreduction;

	// Define the dictionaries
	OpenSMOKE::OpenSMOKE_DictionaryManager dictionaries;
	dictionaries.ReadDictionariesFromFile(input_file_name_);
	dictionaries(main_dictionary_name_).SetGrammar(grammar_staticreduction);

	// Kinetic folder
	boost::filesystem::path path_kinetics_folder;
	if (dictionaries(main_dictionary_name_).CheckOption("@KineticsFolder") == true)
	{
		dictionaries(main_dictionary_name_).ReadPath("@KineticsFolder", path_kinetics_folder);
		OpenSMOKE::CheckKineticsFolder(path_kinetics_folder);
	}

	// Kinetic file
	boost::filesystem::path chemkin_kinetics_file;
	if (dictionaries(main_dictionary_name_).CheckOption("@Kinetics") == true)
		dictionaries(main_dictionary_name_).ReadPath("@Kinetics", chemkin_kinetics_file);

	// Thermodynamic file
	boost::filesystem::path chemkin_thermodynamics_file;
	if (dictionaries(main_dictionary_name_).CheckOption("@Thermodynamics") == true)
		dictionaries(main_dictionary_name_).ReadPath("@Thermodynamics", chemkin_thermodynamics_file);

	// Thermodynamic file
	bool iTransport = false;
	boost::filesystem::path chemkin_transport_file;
	if (dictionaries(main_dictionary_name_).CheckOption("@Transport") == true)
	{
		iTransport = true;
		dictionaries(main_dictionary_name_).ReadPath("@Transport", chemkin_transport_file);
	}

	// Input file
	boost::filesystem::path path_xml_input_file;
	if (dictionaries(main_dictionary_name_).CheckOption("@XMLInput") == true)
		dictionaries(main_dictionary_name_).ReadPath("@XMLInput", path_xml_input_file);

	// Output folder
	boost::filesystem::path path_output_folder;
	if (dictionaries(main_dictionary_name_).CheckOption("@Output") == true)
	{
		dictionaries(main_dictionary_name_).ReadPath("@Output", path_output_folder);

		if (!boost::filesystem::exists(path_output_folder))
			OpenSMOKE::CreateDirectory(path_output_folder);
	}

	// DRG Analysis
	bool iDRG = false;
	if (dictionaries(main_dictionary_name_).CheckOption("@DRG") == true)
		dictionaries(main_dictionary_name_).ReadBool("@DRG", iDRG);

	// DRG-EP Analysis
	bool iDRGEP = false;
	if (dictionaries(main_dictionary_name_).CheckOption("@DRGEP") == true)
		dictionaries(main_dictionary_name_).ReadBool("@DRGEP", iDRGEP);

	bool iTestingNeuralNetwork = false;
	if (dictionaries(main_dictionary_name_).CheckOption("@TestingNeuralNetwork") == true)
		dictionaries(main_dictionary_name_).ReadBool("@TestingNeuralNetwork", iTestingNeuralNetwork);

	// Threshold
	double epsilon = 0.01;
	if (dictionaries(main_dictionary_name_).CheckOption("@Epsilon") == true)
		dictionaries(main_dictionary_name_).ReadDouble("@Epsilon", epsilon);

	// Threshold
	double retained_threshold = 0.01;
	if (dictionaries(main_dictionary_name_).CheckOption("@RetainedThreshold") == true)
		dictionaries(main_dictionary_name_).ReadDouble("@RetainedThreshold", retained_threshold);

	// Temperature threshold 
	double T_Threshold = 310.;
	if (dictionaries(main_dictionary_name_).CheckOption("@TemperatureThreshold") == true)
	{
		std::string units;
		dictionaries(main_dictionary_name_).ReadMeasure("@TemperatureThreshold", T_Threshold, units);
		if (units == "K")			T_Threshold *= 1.;
		else if (units == "C")	T_Threshold += 273.15;
		else OpenSMOKE::FatalErrorMessage("Wrong temperature units");
	}

	// Pressure 
	double P = 101325.;
	if (dictionaries(main_dictionary_name_).CheckOption("@Pressure") == true)
	{
		std::string units;
		dictionaries(main_dictionary_name_).ReadMeasure("@Pressure", P, units);
		if (units == "Pa")			P *= 1.;
		else if (units == "atm")	P *= 101325.;
		else if (units == "bar")	P *= 100000.;
		else OpenSMOKE::FatalErrorMessage("Wrong Pressure units");
	}

	// List of key species
	std::vector<std::string> key_species;
	if (dictionaries(main_dictionary_name_).CheckOption("@KeySpecies") == true)
		dictionaries(main_dictionary_name_).ReadOption("@KeySpecies", key_species);


	// Applying Static Reduction
	{
		std::cout << "Applying static reduction" << std::endl;

		OpenSMOKE::ThermodynamicsMap_CHEMKIN*	thermodynamicsMap;
		OpenSMOKE::KineticsMap_CHEMKIN* 		kineticsMap;

		// Reading thermodynamic and kinetic files
		{ 
			rapidxml::xml_document<> doc;
			std::vector<char> xml_string;
			OpenSMOKE::OpenInputFileXML(doc, xml_string, path_kinetics_folder / "kinetics.xml");

			double tStart = OpenSMOKE::OpenSMOKEGetCpuTime();

			thermodynamicsMap = new OpenSMOKE::ThermodynamicsMap_CHEMKIN(doc);
			kineticsMap = new OpenSMOKE::KineticsMap_CHEMKIN(*thermodynamicsMap, doc);

			double tEnd = OpenSMOKE::OpenSMOKEGetCpuTime();
			std::cout << " * Time to read XML file: " << tEnd - tStart << std::endl;
		}

		// In case of shared memory calculations
		// Adjust number of threads if parametric analysis is required
				
		#if defined(_OPENMP)

		// Indicates that the number of threads available in subsequent parallel region can be adjusted by the run time.
		// A value that indicates if the number of threads available in subsequent parallel region can be adjusted by 
		// the runtime. If nonzero, the runtime can adjust the number of threads, if zero, the runtime will not 
		// dynamically adjust the number of threads.
		omp_set_dynamic(0);
		omp_set_num_threads(number_threads);

		// Info on video
		std::cout << "-----------------------------------------------------------------------" << std::endl;
		std::cout << "              Parameteric Analysis using OpenMP threading              " << std::endl;
		std::cout << "-----------------------------------------------------------------------" << std::endl;
		std::cout << " * Number of CPUs available               = " << omp_get_num_procs() << std::endl;
		std::cout << " * Maximum number of threads available    = " << omp_get_max_threads() << std::endl;
		std::cout << " * Number of threads selected by the user = " << number_threads << std::endl;
		std::cout << " * Number of current threads              = " << omp_get_num_threads() << std::endl;
		std::cout << "-----------------------------------------------------------------------" << std::endl;

		// Read thermodynamics and kinetics maps
		std::vector<OpenSMOKE::ThermodynamicsMap_CHEMKIN*>	vector_thermodynamicsMapXML(number_threads);
		std::vector<OpenSMOKE::KineticsMap_CHEMKIN*>		vector_kineticsMapXML(number_threads);

		for (int j = 0; j < number_threads; j++)
		{
			rapidxml::xml_document<> doc;
			std::vector<char> xml_string;
			OpenSMOKE::OpenInputFileXML(doc, xml_string, path_kinetics_folder / "kinetics.xml");

			double tStart = OpenSMOKE::OpenSMOKEGetCpuTime();
			vector_thermodynamicsMapXML[j] = new OpenSMOKE::ThermodynamicsMap_CHEMKIN(doc);
			vector_kineticsMapXML[j] = new OpenSMOKE::KineticsMap_CHEMKIN(*vector_thermodynamicsMapXML[j], doc);
			double tEnd = OpenSMOKE::OpenSMOKEGetCpuTime();
			std::cout << "Time to read XML file: " << tEnd - tStart << std::endl;
		}

		#endif


		std::cout << "Reading input data..." << std::endl;

		bool read_pca_data = false;
		unsigned int nclusters = 0;					// total number of clusters
		unsigned int ndata = 0;						// total number of items
		unsigned int ns = 0;						// number of original species
		unsigned int nf = 0;						// number of filtered species
		unsigned int npca = 0;						// number of principal components
		unsigned int nretspecies = 0;				// number of retained species
		Eigen::VectorXi group;						// group to which each item belongs
		Eigen::VectorXd csi;						// mixture fraction
		Eigen::VectorXd T;							// temperature [K]
		Eigen::MatrixXd omega;						// mass fractions (original)
		std::vector< std::vector<int> > belonging;	// list of items belonging to each group
		Eigen::VectorXd mu;							// means of filtered variables
		Eigen::VectorXd sigma;						// std deviations of filtered variables
		Eigen::MatrixXd w;							// PCA weights
		Eigen::MatrixXd pca;						// PCA data
		std::vector<int> listretspecies;			// list of retained species (1-based)

		{
			std::cout << "Opening XML file..." << std::endl;

			// Open the XML file
			rapidxml::xml_document<> doc;
			std::vector<char> xml_string;
			OpenSMOKE::OpenInputFileXML(doc, xml_string, path_xml_input_file);
			rapidxml::xml_node<>* opensmoke_node = doc.first_node("opensmoke");

			std::cout << "Reading classes..." << std::endl;

			// Read number of clusters
			rapidxml::xml_node<>* nclusters_node = opensmoke_node->first_node("classes");
			try
			{
				nclusters = boost::lexical_cast<unsigned int>(boost::trim_copy(std::string(nclusters_node->value())));
				std::cout << " * Number of clusters: " << nclusters << std::endl;
			}
			catch (...)
			{
				OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the number of clusters.");
			}

			// Read number of items
			{
				rapidxml::xml_node<>* ndata_node = opensmoke_node->first_node("items");
				try
				{
					ndata = boost::lexical_cast<unsigned int>(boost::trim_copy(std::string(ndata_node->value())));
					std::cout << " * Number of items: " << ndata << std::endl;
				}
				catch (...)
				{
					OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the number of items.");
				}
			}

			// Read number of original species
			{
				rapidxml::xml_node<>* noriginalcomponents_node = opensmoke_node->first_node("original-components");
				try
				{
					ns = boost::lexical_cast<unsigned int>(boost::trim_copy(std::string(noriginalcomponents_node->value())));
					ns -= 1; // one of the original components is always the temperature
					if (ns != thermodynamicsMap->NumberOfSpecies())
						OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "It seems that the preprocessed file is not consistent with the kinetic mechanism you are using, since the number of species is not the same");
					std::cout << " * Number of original species: " << ns << std::endl;
				}
				catch (...)
				{
					OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the number of original components.");
				}
			}

			// Read number of filtered species
			{
				rapidxml::xml_node<>* nfilteredcomponents_node = opensmoke_node->first_node("filtered-components");
				try
				{
					nf = boost::lexical_cast<unsigned int>(boost::trim_copy(std::string(nfilteredcomponents_node->value())));
					nf -= 1; // one of the filtered components is always the temperature
					std::cout << " * Number of filtered species: " << nf << std::endl;
					std::cout << " * Number of removed species:  " << ns-nf << std::endl;
				}
				catch (...)
				{
					OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the number of filteed components.");
				}
			}

			if (read_pca_data == true)
			{
				// Read number of principal components
				{
					rapidxml::xml_node<>* npca_node = opensmoke_node->first_node("principal-components");
					try
					{
						npca = boost::lexical_cast<unsigned int>(boost::trim_copy(std::string(npca_node->value())));
						std::cout << " * Number of principal components: " << npca << std::endl;
					}
					catch (...)
					{
						OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the number principal components.");
					}
				}
			}

			// Read number retained species 
			{
				rapidxml::xml_node<>* nretspecies_node = opensmoke_node->first_node("number-retained-species");
				try
				{
					nretspecies = boost::lexical_cast<unsigned int>(boost::trim_copy(std::string(nretspecies_node->value())));
					std::cout << " * Number of retained species: " << nretspecies << std::endl;
				}
				catch (...)
				{
					OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the number of retained species.");
				}
			}

			// Read list retained species 
			if (nretspecies != thermodynamicsMap->NumberOfSpecies())
			{
				rapidxml::xml_node<>* listretspecies_node = opensmoke_node->first_node("list-retained-species");
				try
				{
					listretspecies.resize(nretspecies);

					std::stringstream fInput;
					fInput << listretspecies_node->value();

					for (unsigned int j = 0; j < nretspecies; j++)
						fInput >> listretspecies[j];	// (1-index based)

					std::cout << std::endl;
					std::cout << "Retained species: " << nretspecies << std::endl;
					for (unsigned int j = 0; j < nretspecies; j++)
						std::cout << thermodynamicsMap->NamesOfSpecies()[listretspecies[j] - 1] << std::endl;
					std::cout << std::endl;
				}
				catch (...)
				{
					OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the list of retained species.");
				}
			}

			group.resize(ndata);
			csi.resize(ndata);
			T.resize(ndata);
			omega.resize(ndata, ns);
			belonging.resize(nclusters);

			if (read_pca_data == true)
			{
				mu.resize(nf + 1);
				sigma.resize(nf + 1);
				w.resize(nf + 1, npca);
				pca.resize(ndata, npca);
			}

			// Read original data
			{
				rapidxml::xml_node<>* dataoriginal_node = opensmoke_node->first_node("data-original");
				try
				{
					std::stringstream fInput;
					fInput << dataoriginal_node->value();

					for (unsigned int j = 0; j < ndata; j++)
					{
						fInput >> group(j);
						fInput >> csi(j);
						fInput >> T(j);

						for (int i = 0; i < ns; i++)
							fInput >> omega(j, i);

						belonging[group(j) - 1].push_back(j);
					}

					std::cout << " * Distribution of items among groups: " << nf << std::endl;
					for (unsigned int j = 0; j < nclusters; j++)
						std::cout << "    + "<<  j << " " << belonging[j].size() << " " << belonging[j].size() / double(ndata)*100. << "%" << std::endl;
				}
				catch (...)
				{
					OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the original data.");
				}
			}

			if (read_pca_data == true)
			{
				// Read mean values of filtered variables
				{
					rapidxml::xml_node<>* mu_node = opensmoke_node->first_node("mu");
					try
					{
						std::stringstream fInput;
						fInput << mu_node->value();

						for (unsigned int j = 0; j < nf + 1; j++)
							fInput >> mu(j);
					}
					catch (...)
					{
						OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading means of filtered variables.");
					}
				}
			
				// Read std deviations of filtered variables
				{
					rapidxml::xml_node<>* sigma_node = opensmoke_node->first_node("sigma");
					try
					{
						std::stringstream fInput;
						fInput << sigma_node->value();

						for (unsigned int j = 0; j < nf + 1; j++)
							fInput >> sigma(j);
					}
					catch (...)
					{
						OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading std deviations of filtered variables.");
					}
				}

				// Read PCA weights
				{
					rapidxml::xml_node<>* w_node = opensmoke_node->first_node("weights");
					try
					{
						std::stringstream fInput;
						fInput << w_node->value();

						for (unsigned int j = 0; j < nf + 1; j++)
							for (unsigned int i = 0; i < npca; i++)
								fInput >> w(j, i);
					}
					catch (...)
					{
						OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading weights of PCA.");
					}
				}

				// Read PCA
				{
					rapidxml::xml_node<>* datapca_node = opensmoke_node->first_node("data-pca");
					try
					{
						std::stringstream fInput;
						fInput << datapca_node->value();

						for (unsigned int j = 0; j < ndata; j++)
						{
							unsigned int dummy;
							fInput >> dummy;
							for (int i = 0; i < npca; i++)
								fInput >> pca(j, i);
						}
					}
					catch (...)
					{
						OpenSMOKE::ErrorMessage("Importing data from MATLAB(R) preprocessing", "Error in reading the PCA data.");
					}
				}
			}

			// Check PCA reconstruction (TOADJUST)
			/*
			{
				Eigen::VectorXd original(nf + 1);
				Eigen::VectorXd transformed(npca);

				for (unsigned int k = 0; k < 1000; k += 5)
				{
					std::cout << "Point " << k << std::endl;

					original(0) = (T(k)-mu(0))/sigma(0);
					for (unsigned int i = 0; i < ns; i++)
						original(i + 1) = (omega(k, i)-mu(i+1))/sigma(i+1);
					transformed = original.transpose()*w;

					for (unsigned int i = 0; i < npca; i++)
						std::cout << transformed(i) << " " << pca(k, i) << " " << transformed(i) - pca(k, i) << std::endl;

					getchar();
				}
			}
			*/
		}

		if (iDRG == true)
		{
			std::cout << "Applying DRG..." << std::endl;

			boost::filesystem::path path_drg_output_folder = path_output_folder / "DRG";
			if (!boost::filesystem::exists(path_drg_output_folder))
				OpenSMOKE::CreateDirectory(path_drg_output_folder);

			// Preparing Analysis
			Eigen::VectorXi number_important_species(ndata);
			Eigen::MatrixXi important_species(ndata, ns); important_species.setZero();
			
			// Analysis
			OpenSMOKE::OpenSMOKEVectorDouble y(ns);
			OpenSMOKE::OpenSMOKEVectorDouble x(ns);
			OpenSMOKE::OpenSMOKEVectorDouble c(ns);

			OpenSMOKE::DRG drg(thermodynamicsMap, kineticsMap);
			for (int j = 0; j < ndata; j++)
			{
				drg.SetKeySpecies(key_species);
				drg.SetEpsilon(epsilon);
				drg.SetTemperatureThreshold(T_Threshold);
				//drg.SetEpsilon(EpsilonDRG(T(j)));		// TODO

				for (int k = 0; k < ns; k++)
					y[k + 1] = omega(j, k);

				double MW;
				thermodynamicsMap->MoleFractions_From_MassFractions(x.GetHandle(), MW, y.GetHandle());

				const double threshold_c = 1.e-14;
				const double cTot = P / PhysicalConstants::R_J_kmol / T(j);
				OpenSMOKE::Product(cTot, x, &c);
				for (int k = 1; k <= ns; k++)
					if (c(k) < threshold_c)	c(k) = 0.;

				drg.Analysis(T(j), P, c);

				number_important_species(j) = drg.number_important_species();
				for (int i = 0; i < number_important_species(j); ++i)
				{
					const unsigned int k = drg.indices_important_species()[i];
					important_species(j, k) = 1;
				}

				if ((j + 1) % 1000 == 1)
					std::cout << j << "/" << ndata << " T: " << T(j) << " Species: " << number_important_species(j) << " Eps: " << epsilon << std::endl;
	//				std::cout << j << "/" << ndata << " T: " << T(j) << " Species: " << number_important_species(j) << " Eps: " << EpsilonDRG(T(j)) << std::endl;
			}

			// Post processing analysis

			// Matrix of presence of species
			PrintMatrixOfPresence(	path_drg_output_folder, ndata, 
									number_important_species, important_species);

			// Error analysis
			Eigen::MatrixXi retained_species;
			PrintErrorAnalysis(	path_drg_output_folder, ndata, nclusters,
								number_important_species,important_species,
								belonging, thermodynamicsMap->NamesOfSpecies(),
								retained_threshold,
								retained_species);

			// Select reactions
			Eigen::MatrixXi retained_reactions;
			SelectImportantReactions(*kineticsMap, retained_species, retained_reactions);

			// Write kinetic mechanisms
			WriteKineticMechanisms(*thermodynamicsMap, *kineticsMap,
									path_drg_output_folder, chemkin_thermodynamics_file, chemkin_kinetics_file,
									retained_species, retained_reactions);

			// Preprocess kinetic mechanisms
			if (iTransport == true)
			{
				PreprocessKineticMechanisms(nclusters, path_drg_output_folder,
											chemkin_thermodynamics_file, 
											chemkin_transport_file);
			}
		}

		if (iDRGEP == true)
		{
			std::cout << "Applying DRGEP..." << std::endl;

			boost::filesystem::path path_drgep_output_folder = path_output_folder / "DRGEP";
			if (!boost::filesystem::exists(path_drgep_output_folder))
				OpenSMOKE::CreateDirectory(path_drgep_output_folder);

			Eigen::VectorXi number_important_species(ndata);
			Eigen::MatrixXi important_species(ndata, ns); important_species.setZero();

			OpenSMOKE::OpenSMOKEVectorDouble y(ns);
			OpenSMOKE::OpenSMOKEVectorDouble x(ns);
			OpenSMOKE::OpenSMOKEVectorDouble c(ns);

			Eigen::VectorXi items_cluster(nclusters);
			items_cluster.setZero();
			for (int jj = 0; jj < nclusters; jj++)
				for (int j = 0; j < ndata; j++)
					if (group(j) == jj+1)
						items_cluster(jj)++;

			std::vector<Eigen::VectorXi> group_cluster(nclusters);
			for (int jj = 0; jj < nclusters; jj++)
				group_cluster[jj].resize(items_cluster(jj));

			items_cluster.setZero();
			for (int jj = 0; jj < nclusters; jj++)
				for (int j = 0; j < ndata; j++)
					if (group(j) == jj + 1)
						group_cluster[jj](items_cluster(jj)++) = j;

			#if defined(_OPENMP)
			if (number_threads > 1)
			{	
				for (int jj = 0; jj < nclusters; jj++)
				{
					#pragma omp parallel num_threads(number_threads)
					{
						const int mytid = omp_get_thread_num();

						OpenSMOKE::OpenSMOKEVectorDouble yy(ns);
						OpenSMOKE::OpenSMOKEVectorDouble xx(ns);
						OpenSMOKE::OpenSMOKEVectorDouble cc(ns);

						OpenSMOKE::DRGEP drgep(vector_thermodynamicsMapXML[omp_get_thread_num()], vector_kineticsMapXML[omp_get_thread_num()]);
						drgep.SetKeySpecies(key_species);
						drgep.SetEpsilon(epsilon);
						drgep.SetTemperatureThreshold(T_Threshold);
						drgep.PrepareKineticGraph(key_species);

						for (int jLocal = 0; jLocal < group_cluster[jj].size(); jLocal++)
						{
							if (jLocal%number_threads == mytid) 
							{
								const int j = group_cluster[jj][jLocal];

								for (int k = 0; k < ns; k++)
									yy[k + 1] = omega(j, k);

								double MW;
								vector_thermodynamicsMapXML[mytid]->MoleFractions_From_MassFractions(xx.GetHandle(), MW, yy.GetHandle());

								const double threshold_c = 1.e-14;
								const double cTot = P / PhysicalConstants::R_J_kmol / T(j);
								OpenSMOKE::Product(cTot, xx, &cc);
								for (int k = 1; k <= ns; k++)
									if (cc[k] < threshold_c)	cc[k] = 0.;

								drgep.Analysis(T(j), P, cc);

								number_important_species(j) = drgep.number_important_species();
								for (int i = 0; i < number_important_species(j); ++i)
								{
									const unsigned int k = drgep.indices_important_species()[i];
									important_species(j, k) = 1;
								}

								if (jLocal == 0)
									std::cout << "Group: " << jj << "/" << nclusters
									<< " T: " << T(j)
									<< " Species: " << number_important_species(j)
									<< " Eps: " << epsilon << std::endl;

							}
						}
					}
				}
			}
			else
			#endif
			{
				OpenSMOKE::DRGEP drgep(thermodynamicsMap, kineticsMap);
				drgep.SetKeySpecies(key_species);
				drgep.SetEpsilon(epsilon);
				drgep.SetTemperatureThreshold(T_Threshold);
				//drgep.SetEpsilon(EpsilonDRGEP(T(j)));	// TODO
				drgep.PrepareKineticGraph(key_species);

				for (int jj = 0; jj < nclusters; jj++)
				{

					for (int jLocal = 0; jLocal < group_cluster[jj].size(); jLocal++)
					{
						const int j = group_cluster[jj][jLocal];

						for (int k = 0; k < ns; k++)
							y[k + 1] = omega(j, k);

						double MW;
						thermodynamicsMap->MoleFractions_From_MassFractions(x.GetHandle(), MW, y.GetHandle());

						const double threshold_c = 1e-14;
						const double cTot = P / PhysicalConstants::R_J_kmol / T(j);
						OpenSMOKE::Product(cTot, x, &c);
						for (int k = 1; k <= ns; k++)
							if (c[k] < threshold_c)	c[k] = 0.;

						drgep.Analysis(T(j), P, c);

						number_important_species(j) = drgep.number_important_species();
						for (int i = 0; i < number_important_species(j); ++i)
						{
							const unsigned int k = drgep.indices_important_species()[i];
							important_species(j, k) = 1;
						}

						if (jLocal == 0)
							std::cout << "Group: " << jj << "/" << nclusters
							<< " T: " << T(j)
							<< " Species: " << number_important_species(j)
							<< " Eps: " << epsilon << std::endl;

					}
				}
			}

			// Post processing analysis

			// Matrix of presence of species
			PrintMatrixOfPresence(	path_drgep_output_folder, ndata,
									number_important_species, important_species);

			// Error analysis
			Eigen::MatrixXi retained_species;
			PrintErrorAnalysis(	path_drgep_output_folder, ndata, nclusters,
								number_important_species, important_species,
								belonging, thermodynamicsMap->NamesOfSpecies(),
								retained_threshold,
								retained_species);

			// Select reactions
			Eigen::MatrixXi retained_reactions;
			SelectImportantReactions(*kineticsMap, retained_species, retained_reactions);

			// Write kinetic mechanisms
			WriteKineticMechanisms(*thermodynamicsMap, *kineticsMap,
									path_drgep_output_folder, chemkin_thermodynamics_file, chemkin_kinetics_file,
									retained_species, retained_reactions);

			// Preprocess kinetic mechanisms
			if (iTransport == true)
			{
				PreprocessKineticMechanisms(nclusters, path_drgep_output_folder,
											chemkin_thermodynamics_file,
											chemkin_transport_file);
			}
		}

		// Testing the network
		/*
		if (iTestingNeuralNetwork == true)
		{
			Eigen::VectorXd y(nclusters);

			int success = 0;
			for (unsigned int i = 0; i < ndata; i++)
			{
				Eigen::VectorXd row = pca.row(i);
				myNeuralNetworkBasedOnPCA(row.data(), y.data());
				
				int k;  
				y.maxCoeff(&k);
				if (k+1 == group(i))
					success++;

				// Example
				if (i % 1000 == 0)
				{
					std::cout << "Testing point: " << i << " - Group: " << group(i) << std::endl;
					std::cout << " * Returned group: " << k + 1 << std::endl;
					std::cout << " * Complete list " << std::endl;
					for (int j = 0; j < nclusters; j++)
						std::cout << j + 1 << " " << y(j) << std::endl;
				}
			}

			std::cout << "Summary" << std::endl;
			std::cout << success << " / " << ndata << " / " << double(success) / double(ndata)*100. << std::endl;
		}
		*/
		
	}

	bool iPostProcessFoam = true;
	if (iPostProcessFoam == true)
	{
		OpenSMOKE::ThermodynamicsMap_CHEMKIN*	thermodynamicsMap;

		// Reading thermodynamic and kinetic files
		{
			rapidxml::xml_document<> doc;
			std::vector<char> xml_string;
			OpenSMOKE::OpenInputFileXML(doc, xml_string, path_kinetics_folder / "kinetics.xml");

			double tStart = OpenSMOKE::OpenSMOKEGetCpuTime();

			thermodynamicsMap = new OpenSMOKE::ThermodynamicsMap_CHEMKIN(doc);

			double tEnd = OpenSMOKE::OpenSMOKEGetCpuTime();
			std::cout << " * Time to read XML file: " << tEnd - tStart << std::endl;
		}

		std::vector<std::string> list_labels;
		list_labels.push_back("zMix");
		list_labels.push_back("T");
		for (unsigned int i = 0; i < thermodynamicsMap->NumberOfSpecies(); i++)
			list_labels.push_back(thermodynamicsMap->NamesOfSpecies()[i]);

		std::vector<std::string> list_times;
		list_times.push_back("0");
		list_times.push_back("0.1");
		list_times.push_back("0.11");
		list_times.push_back("0.12");
		list_times.push_back("0.13");
		list_times.push_back("0.14");
		list_times.push_back("0.15");

		const unsigned int skip_lines = 22;
		const unsigned int npoints = 21334;
		const double T_threshold = 299.;

		boost::filesystem::path global_folder_name = "C:/Users/acuoci/Desktop/PCI 2018/Simulations/Unsteady/Detailed/";
		for (unsigned int k = 0; k < list_times.size(); k++)
		{
			std::cout << "Processing time: " << list_times[k] << std::endl;

			boost::filesystem::path folder_name = global_folder_name / list_times[k];
			
			Eigen::MatrixXd values(npoints, list_labels.size());
			for (unsigned int i = 0; i < list_labels.size(); i++)
			{
				boost::filesystem::path file_name = folder_name / list_labels[i];
				std::ifstream fInput(file_name.c_str(), std::ios::in);

				std::string dummy;
				for(unsigned j=0;j<skip_lines;j++)
					std::getline(fInput, dummy);

				for (unsigned j = 0; j < npoints; j++)
					fInput >> values(j, i);

				fInput.close();
			}

			boost::filesystem::path file_name = global_folder_name / ("output." + list_times[k]);
			std::ofstream fOutput(file_name.c_str(), std::ios::out);
			fOutput << "Data from OpenFOAM simulation" << std::endl;
			for (unsigned j = 0; j < npoints; j++)
			{
				if (values(j, 1) > T_threshold)
				{
					fOutput << std::setw(3) << std::fixed << 0;	// dummy col
					fOutput << std::setw(3) << std::fixed << 0;	// dummy col
					fOutput << std::setw(3) << std::fixed << 0;	// dummy col
					for (unsigned int i = 0; i < list_labels.size(); i++)
						fOutput << std::setw(15) << std::scientific << std::setprecision(6) << values(j, i);
					fOutput << std::endl;
				}
			}
			fOutput.close();
		}

	}

	std::cout << "Press enter to exit..." << std::endl;
	getchar();
	return 0;

}

void PrintMatrixOfPresence(	const boost::filesystem::path& path_output_folder, 
							const unsigned int ndata, 
							const Eigen::VectorXi& number_important_species,
							const Eigen::MatrixXi& important_species)
{
	boost::filesystem::path filename = path_output_folder / "presence.out";
	std::ofstream fOut(filename.c_str(), std::ios::out);
	for (int j = 0; j < ndata; j++)
	{
		fOut << std::setw(5) << std::left << number_important_species(j);
		for (int k = 0; k < important_species.cols(); k++)
			fOut << std::setw(2) << std::left << important_species(j, k);
		fOut << std::endl;
	}
	fOut.close();
}

void PrintErrorAnalysis(	const boost::filesystem::path& path_output_folder,
							const unsigned int ndata, const unsigned int nclusters,
							const Eigen::VectorXi& number_important_species,
							const Eigen::MatrixXi& important_species,
							const std::vector< std::vector<int> > belonging,
							const std::vector<std::string> species_names,
							const double retained_threshold,
							Eigen::MatrixXi& retained_species)
{
	const unsigned int ns = important_species.cols();

	Eigen::MatrixXi sums(nclusters, ns); sums.setZero();
	Eigen::MatrixXd ratios(nclusters, ns); ratios.setZero();
	Eigen::MatrixXd errors(nclusters, ns); errors.setZero();
	retained_species.resize(nclusters, ns); retained_species.setZero();

	Eigen::VectorXi retained(nclusters); retained.setZero();
	Eigen::VectorXi retained_000(nclusters); retained_000.setZero();
	Eigen::VectorXi retained_001(nclusters); retained_001.setZero();
	Eigen::VectorXi retained_002(nclusters); retained_002.setZero();
	Eigen::VectorXi retained_005(nclusters); retained_005.setZero();
	Eigen::VectorXi retained_010(nclusters); retained_010.setZero();
	Eigen::VectorXi retained_020(nclusters); retained_020.setZero();
	
	// Analysis of errors
	for (int i = 0; i < nclusters; i++)
	{
		for (int j = 0; j < belonging[i].size(); j++)
		{
			const int g = belonging[i][j];
			for (int k = 0; k < ns; k++)
				sums(i, k) += important_species(g, k);
		}

		for (int k = 0; k < ns; k++)
			ratios(i, k) = double(sums(i, k)) / double(belonging[i].size());

		for (int k = 0; k < ns; k++)
			if (ratios(i, k) > 0.) errors(i, k) = std::fabs(1. - ratios(i, k));

		// Retain species only if the frequency of presence is larger than a threshold
		for (int k = 0; k < ns; k++)
		{
			if (ratios(i, k) > 0.00)	retained_000(i)++;
			if (ratios(i, k) > 0.01)	retained_001(i)++;
			if (ratios(i, k) > 0.02)	retained_002(i)++;
			if (ratios(i, k) > 0.05)	retained_005(i)++;
			if (ratios(i, k) > 0.10)	retained_010(i)++;
			if (ratios(i, k) > 0.20)	retained_020(i)++;
			
			if (ratios(i, k) > retained_threshold)
			{
				retained(i)++;
				retained_species(i, k) = 1;
			}
		}
	}

	{
		boost::filesystem::path filename = path_output_folder / "sums.out";
		std::ofstream fSums(filename.c_str(), std::ios::out);

		fSums << std::left << std::setw(7) << "#";
		fSums << std::left << std::setw(12) << "Items";
		
		unsigned int count = 3;
		for (int k = 0; k < ns; k++)
		{
			std::stringstream index; index << count++;
			std::string label = species_names[k] + "("+ index.str() + ")";
			fSums << std::left << std::setw(20) << label;
		}
		fSums << std::endl;
		
		// Local
		for (int i = 0; i < nclusters; i++)
		{
			fSums << std::left << std::setw(7) << i;
			fSums << std::left << std::setw(12) << belonging[i].size();
			for (int k = 0; k < ns; k++)
				fSums << std::left << std::setw(20) << sums(i, k);
			fSums << std::endl;
		}

		// Global
		{
			fSums << std::setw(7) << std::left << "Tot";
			fSums << std::setw(12) << std::left << ndata;
			for (int k = 0; k < ns; k++)
			{
				unsigned int sum = 0;
				for (int i = 0; i < nclusters; i++)
					sum += sums(i, k);
				fSums << std::setw(20) << std::left << sum;
			}
			fSums << std::endl;
		}

		fSums.close();
	}

	// Errors
	{
		Eigen::VectorXd sum_errors(nclusters); sum_errors.setZero();
		for (int i = 0; i < nclusters; i++)
		{
			for (int k = 0; k < ns; k++)
				sum_errors(i) += errors(i, k);
			sum_errors(i) *= 100. / double(ns);
		}

		{
			boost::filesystem::path filename = path_output_folder / "errors.out";
			std::ofstream fErrors(filename.c_str(), std::ios::out);

			fErrors << std::left << std::fixed << std::setw(5) << "#";
			fErrors << std::left << std::fixed << std::setw(12) << "error(%)";
			fErrors << std::left << std::fixed << std::setw(7) << "S(0%)";
			fErrors << std::left << std::fixed << std::setw(7) << "S(1%)";
			fErrors << std::left << std::fixed << std::setw(7) << "S(2%)";
			fErrors << std::left << std::fixed << std::setw(7) << "S(5%)";
			fErrors << std::left << std::fixed << std::setw(7) << "S(10%)";
			fErrors << std::left << std::fixed << std::setw(7) << "S(20%)";
			fErrors << std::endl;

			for (int i = 0; i < nclusters; i++)
			{
				fErrors << std::left << std::fixed << std::setw(5) << i;
				fErrors << std::left << std::fixed << std::setprecision(3) << std::setw(12) << sum_errors(i);
				fErrors << std::left << std::fixed << std::setw(7) << retained_000(i);
				fErrors << std::left << std::fixed << std::setw(7) << retained_001(i);
				fErrors << std::left << std::fixed << std::setw(7) << retained_002(i);
				fErrors << std::left << std::fixed << std::setw(7) << retained_005(i);
				fErrors << std::left << std::fixed << std::setw(7) << retained_010(i);
				fErrors << std::left << std::fixed << std::setw(7) << retained_020(i);
				fErrors << std::endl;
			}
			fErrors.close();
		}

		std::cout << "Errors(%)" << std::endl;
		for (int i = 0; i < nclusters; i++)
		{
			std::cout << std::left << std::fixed << std::setw(5) << i;
			std::cout << std::left << std::fixed << std::setprecision(3) << std::setw(12) << sum_errors(i);
			std::cout << std::left << std::fixed << std::setw(7) << retained_000(i);
		}
	}

	// Similarities between groups
	{
		std::cout << " * Writing similarity file..." << std::endl;

		Eigen::MatrixXd similarities(nclusters, nclusters);
		similarities.setZero();
		{
			for (int i = 0; i < nclusters; i++)
				for (int j = 0; j < nclusters; j++)
				{
					double sum = 0.;
					for (int k = 0; k < ns; k++)
						sum += std::fabs(retained_species(i, k) - retained_species(j, k));
					sum /= double(ns);
					similarities(i, j) = 1. - sum;
				}
		}

		boost::filesystem::path filename = path_output_folder / "similarities.out";
		std::ofstream fSimilarities(filename.c_str(), std::ios::out);

		fSimilarities << std::setw(12) << std::left << "Clusters";
		for (int i = 0; i < nclusters; i++)
			fSimilarities << std::setw(6) << std::left << i;
		fSimilarities << std::endl;

		for (int i = 0; i < nclusters; i++)
		{
			fSimilarities << std::left << std::fixed << std::setw(12) << i;
			for (int j = 0; j < nclusters; j++)
				fSimilarities << std::setw(6) << std::setprecision(3) << std::left << similarities(i, j);
			fSimilarities << std::endl;
		}

		fSimilarities.close();
	}

	// Uniformity coefficients
	{
		std::cout << " * Writing similarity file..." << std::endl;

		boost::filesystem::path filename = path_output_folder / "uniformity.out";
		std::ofstream fUniformity(filename.c_str(), std::ios::out);

		fUniformity << std::setw(10) << std::setprecision(3)  << std::left << "Cluster";
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << "Samples";
		fUniformity << std::setw(10) << std::setprecision(3)  << std::left << "Species";
		fUniformity << std::setw(10) << std::setprecision(6)  << std::left << "lambda";
		fUniformity << std::endl;
		
		Eigen::VectorXi nspecies(nclusters); nspecies.setZero();
		Eigen::VectorXd lambda(nclusters); lambda.setZero();

		for (int i = 0; i < nclusters; i++)
		{
			// Participation index
			Eigen::VectorXd x(ns); x.setZero();
			for (int j = 0; j < belonging[i].size(); j++)
			{
				const int g = belonging[i][j];
				for (int k = 0; k < ns; k++)
					x(k) += important_species(g, k);
			}
			
			if (belonging[i].size() != 0)
				x /= static_cast<double>(belonging[i].size());

			// Uniformity coefficient
			for (int k = 0; k < ns; k++)
			{
				if (x(k) != 0)
				{
					nspecies(i)++;
					lambda(i) += std::pow(x(k) - 1., 2.);
				}
			}

			lambda(i) = std::sqrt(lambda(i));
			if (nspecies(i) != 0)
				lambda(i) /= static_cast<double>(nspecies(i));

			fUniformity << std::setw(10)  << std::setprecision(3) << std::left << i;
			fUniformity << std::setw(10) << std::setprecision(3) << std::left << belonging[i].size();
			fUniformity << std::setw(10)  << std::setprecision(3) << std::left << nspecies(i);
			fUniformity << std::setw(10)  << std::setprecision(6) << std::left << std::fixed << lambda(i);
			fUniformity << std::endl;
		}

		double mean_nspecies = 0.;
		double mean_lambda = 0.;
		unsigned int n_feasible = 0;
		unsigned int min_nspecies = ns;
		double min_lambda = 1.;
		double sigma2_nspecies = 0.;
		double sigma2_lambda = 0.;
		for (int i = 0; i < nclusters; i++)
		{
			if (nspecies(i) != 0)
			{
				mean_nspecies += nspecies(i);
				mean_lambda += lambda(i);

				if (nspecies(i) < min_nspecies)
					min_nspecies = nspecies(i);

				if (lambda(i) < min_lambda)
					min_lambda = lambda(i);

				sigma2_nspecies += nspecies(i)*nspecies(i);
				sigma2_lambda += lambda(i)*lambda(i);

				n_feasible++;
			}
		}
		mean_nspecies /= static_cast<double>(n_feasible);
		mean_lambda /= static_cast<double>(n_feasible);
		sigma2_nspecies -= n_feasible*mean_nspecies*mean_nspecies;
		sigma2_nspecies /= static_cast<double>(n_feasible);
		sigma2_lambda -= n_feasible*mean_lambda*mean_lambda;
		sigma2_lambda /= static_cast<double>(n_feasible);
		const double sigma_nspecies = std::sqrt(sigma2_nspecies);
		const double sigma_lambda = std::sqrt(sigma2_lambda);

		
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << "Zeros";
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << ndata;
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << nclusters-n_feasible;
		fUniformity << std::setw(10) << std::setprecision(6) << std::left << std::fixed << (nclusters - n_feasible)/static_cast<double>(nclusters);
		fUniformity << std::endl;

		fUniformity << std::setw(10) << std::setprecision(3) << std::left << "Mean";
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << ndata;
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << mean_nspecies;
		fUniformity << std::setw(10) << std::setprecision(6) << std::left << std::fixed << mean_lambda;
		fUniformity << std::endl;

		fUniformity << std::setw(10) << std::setprecision(3) << std::left << "Sigma";
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << ndata;
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << sigma_nspecies;
		fUniformity << std::setw(10) << std::setprecision(6) << std::left << std::fixed << sigma_lambda;
		fUniformity << std::endl;

		fUniformity << std::setw(10) << std::setprecision(3) << std::left << "Min";
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << ndata;
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << min_nspecies;
		fUniformity << std::setw(10) << std::setprecision(6) << std::left << std::fixed << min_lambda;
		fUniformity << std::endl;

		fUniformity << std::setw(10) << std::setprecision(3) << std::left << "Max";
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << ndata;
		fUniformity << std::setw(10) << std::setprecision(3) << std::left << nspecies.maxCoeff();
		fUniformity << std::setw(10) << std::setprecision(6) << std::left << std::fixed << lambda.maxCoeff();
		fUniformity << std::endl;

		fUniformity.close();
	}
}

void SelectImportantReactions(OpenSMOKE::KineticsMap_CHEMKIN& kineticsMap,
	Eigen::MatrixXi& retained_species,
	Eigen::MatrixXi& retained_reactions)
{
	std::cout << " * Selecting important reactions..." << std::endl;

	unsigned int nclusters = retained_species.rows();
	unsigned int ns = retained_species.cols();
	unsigned int nr = kineticsMap.NumberOfReactions();
	Eigen::SparseMatrix<int> delta_sparse(ns, nr);

	{
		// Build a full matrix of net stoichiometric coefficients nu = nuB - nuF
		Eigen::MatrixXi nu(nr, ns);
		{
			// Be careful: eigen vectors and matrices are 0-index based
			// Be careful: if the kinetic scheme is large, this matrix, since it is full, can be very memory expensive
			//             Example: 10^3 species, 10^4 reactions = size of the matrix 10^7 elements!
			//             This is the reason why we store stoichiometric matrices in sparse format.
			//             Of course te advantage of having a full matrix, is that you access the elements directly, without
			//             using iterators and pointers, as reported above
			nu.setZero();

			// Loop over all the reactions (product side)
			for (int k = 0; k < kineticsMap.stoichiometry().stoichiometric_matrix_products().outerSize(); ++k)
			{
				// Loop over all the non-zero stoichiometric coefficients (product side) of reaction k
				for (Eigen::SparseMatrix<double>::InnerIterator it(kineticsMap.stoichiometry().stoichiometric_matrix_products(), k); it; ++it)
				{
					nu(it.row(), it.col()) = 1;
				}
			}

			// Loop over all the reactions (product side)
			for (int k = 0; k < kineticsMap.stoichiometry().stoichiometric_matrix_reactants().outerSize(); ++k)
			{
				// Loop over all the non-zero stoichiometric coefficients (product side) of reaction k
				for (Eigen::SparseMatrix<double>::InnerIterator it(kineticsMap.stoichiometry().stoichiometric_matrix_reactants(), k); it; ++it)
				{
					nu(it.row(), it.col()) = 1;
				}
			}
		}

		// Sparse delta matrix, 1 means the species is involved in the reaction, (NR x NS)
		{
			typedef Eigen::Triplet<double> T;
			std::vector<T> tripletList;
			tripletList.reserve(nr * 4);
			for (unsigned int i = 0; i < nr; i++)
				for (unsigned int j = 0; j < ns; j++)
				{
					if (nu(i, j) != 0)
						tripletList.push_back(T(j, i, 1));
				}

			delta_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
		}
	}

	// Important reactions
	retained_reactions.resize(nclusters, kineticsMap.NumberOfReactions());
	retained_reactions.setConstant(1);
	for (unsigned int i = 0; i < nclusters; i++)
	{
		for (int k = 0; k < delta_sparse.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<int>::InnerIterator it(delta_sparse, k); it; ++it)
			{
				if (retained_species(i, it.row()) == 0)
				{
					retained_reactions(i, k) = 0;
				}
			}
		}
	}

	// Number of retained reactions
	for (unsigned int i = 0; i < nclusters; i++)
	{
		const unsigned int number_retained_reactions = retained_reactions.row(i).sum();
		std::cout << i << " " << number_retained_reactions << std::endl;
	}

	// Include (if needed) third-body species
	{
		std::vector<unsigned int> third_body_reactions = kineticsMap.IndicesOfThirdbodyReactions();
		const std::vector< std::vector<unsigned int> > third_body_species = kineticsMap.IndicesOfThirdbodySpecies();

		for (unsigned int i = 0; i < nclusters; i++)
		{
			// First loop
			for (unsigned int k = 0; k < third_body_reactions.size(); k++)
			{
				unsigned int index_reaction = third_body_reactions[k] - 1;
				if (retained_reactions(i, index_reaction) == 1)
				{
					std::cout << "Looking for third body reaction: " << index_reaction << std::endl;
					for (unsigned int j = 0; j < third_body_species[k].size(); j++)
					{
						unsigned int index_species = third_body_species[k][j]-1;
						if (retained_species(i, index_species) == 0)
						{
							std::cout << "Adding species: " << index_species << std::endl;
							retained_species(i, index_species) = 1;
						}
					}
				}
			}

			// Falloff reactions
			for (unsigned int k = 0; k < kineticsMap.NumberOfFallOffReactions(); k++)
			{
				const unsigned int index_reaction = kineticsMap.IndicesOfFalloffReactions()[k] - 1;

				if (retained_reactions(i, index_reaction) == 1)
				{
					std::cout << "Looking for fall-off reaction: " << index_reaction << std::endl;

					if (kineticsMap.FallOffIndexOfSingleThirdbodySpecies()[k] == 0)
					{
						for (unsigned int j = 0; j < kineticsMap.FallOffIndicesOfThirdbodySpecies()[k].size(); j++)
						{
							unsigned int index_species = kineticsMap.FallOffIndicesOfThirdbodySpecies()[k][j] - 1;
							if (retained_species(i, index_species) == 0)
							{
								std::cout << "Adding species: " << index_species << std::endl;
								retained_species(i, index_species) = 1;
							}
						}
					}
				}
			}
		}
	}
}

void WriteKineticMechanisms(const OpenSMOKE::ThermodynamicsMap_CHEMKIN& thermodynamicsMap,
							const OpenSMOKE::KineticsMap_CHEMKIN& kineticsMap, 
							const boost::filesystem::path& path_output_folder,
							const boost::filesystem::path& chemkin_thermodynamic_file,
							const boost::filesystem::path& chemkin_kinetics_file,
							const Eigen::MatrixXi& retained_species,
							const Eigen::MatrixXi& retained_reactions)
{
	std::cout << " * Writing mechanisms..." << std::endl;

	boost::filesystem::path path_folder = path_output_folder / "Mechanisms";
	if (!boost::filesystem::exists(path_folder))
		OpenSMOKE::CreateDirectory(path_folder);

	// Log file
	std::ofstream flog;
	{
		boost::filesystem::path file_name = path_folder / "log";
		flog.open(file_name.c_str(), std::ios::out);
		flog.setf(std::ios::scientific);
	}
	
	//Preprocessing kinetic file
	typedef OpenSMOKE::ThermoReader< OpenSMOKE::ThermoReaderPolicy_CHEMKIN< OpenSMOKE::ThermoPolicy_CHEMKIN > > ThermoReader_CHEMKIN;
	ThermoReader_CHEMKIN* thermoreader;

	// Reading thermodynamic database
	thermoreader = new OpenSMOKE::ThermoReader< OpenSMOKE::ThermoReaderPolicy_CHEMKIN< OpenSMOKE::ThermoPolicy_CHEMKIN > >;
	thermoreader->ReadFromFile(chemkin_thermodynamic_file.string());

	//Preprocessing Chemical Kinetics
	typedef OpenSMOKE::PreProcessorKinetics< OpenSMOKE::PreProcessorKineticsPolicy_CHEMKIN<OpenSMOKE::ReactionPolicy_CHEMKIN> > PreProcessorKinetics_CHEMKIN;
	PreProcessorKinetics_CHEMKIN* preprocessor_kinetics;
	preprocessor_kinetics = new PreProcessorKinetics_CHEMKIN(flog);
	preprocessor_kinetics->ReadFromASCIIFile(chemkin_kinetics_file.string());

	// Preprocessing the thermodynamics
	typedef OpenSMOKE::Species< OpenSMOKE::ThermoPolicy_CHEMKIN, OpenSMOKE::TransportPolicy_CHEMKIN > SpeciesCHEMKIN;
	typedef OpenSMOKE::PreProcessorSpecies< OpenSMOKE::PreProcessorSpeciesPolicy_CHEMKIN_WithoutTransport<SpeciesCHEMKIN> > PreProcessorSpecies_CHEMKIN_WithoutTransport;
	PreProcessorSpecies_CHEMKIN_WithoutTransport* preprocessor_species_without_transport;
	preprocessor_species_without_transport = new PreProcessorSpecies_CHEMKIN_WithoutTransport(*thermoreader, *preprocessor_kinetics, flog);
	CheckForFatalError(preprocessor_species_without_transport->Setup());
	
	// Read kinetics from file
	CheckForFatalError(preprocessor_kinetics->ReadKineticsFromASCIIFile(preprocessor_species_without_transport->AtomicTable()));

	delete thermoreader;
	delete preprocessor_species_without_transport;
	
	std::cout << " * Writing mechanisms..." << std::endl;
	unsigned int nclusters = retained_species.rows();
	unsigned int ns = retained_species.cols();
	for (unsigned int i = 0; i < nclusters; i++)
	{
		std::vector<bool> is_reduced_species(ns);
		for (unsigned int k = 0; k<ns; k++)
			is_reduced_species[k] = true;

		std::stringstream label; label << i;
		std::string name = "kinetics." + label.str() + ".CKI";
		boost::filesystem::path filename = path_folder / name;
		
		std::ofstream fKinetics(filename.c_str(), std::ios::out);

		fKinetics << "ELEMENTS" << std::endl;
		for (unsigned int k = 0; k < thermodynamicsMap.elements().size(); k++)
			fKinetics << thermodynamicsMap.elements()[k] << std::endl;
		fKinetics << "END" << std::endl;
		fKinetics << std::endl;

		fKinetics << "SPECIES" << std::endl;
		unsigned int count = 0;
		for (unsigned int k = 0; k < ns; k++)
		{
			if (retained_species(i, k) == 1)
			{
				count++;
				fKinetics << thermodynamicsMap.NamesOfSpecies()[k] << "  ";
				if (count % 6 == 0)	fKinetics << std::endl;
			}
		}
		fKinetics << std::endl;
		fKinetics << "END" << std::endl;
		fKinetics << std::endl;

		fKinetics << "REACTIONS" << std::endl;
		
		for (unsigned int k = 0; k < preprocessor_kinetics->reactions().size(); k++)
		{
			if (retained_reactions(i,k) == 1)
			{
				std::stringstream reaction_data;
				std::string reaction_string;
				preprocessor_kinetics->reactions()[k].GetReactionStringCHEMKIN(thermodynamicsMap.NamesOfSpecies(), reaction_data, is_reduced_species);
				fKinetics << reaction_data.str();
			}
		}
		fKinetics << "END" << std::endl;
		fKinetics << std::endl;

		fKinetics.close();
	}
}

void PreprocessKineticMechanisms(
	const unsigned int nclusters,
	const boost::filesystem::path& path_output_folder,
	const boost::filesystem::path& chemkin_thermodynamic_file,
	const boost::filesystem::path& chemkin_transport_file)
{
	typedef OpenSMOKE::Species< OpenSMOKE::ThermoPolicy_CHEMKIN, OpenSMOKE::TransportPolicy_CHEMKIN > SpeciesCHEMKIN;
	typedef OpenSMOKE::PreProcessorSpecies< OpenSMOKE::PreProcessorSpeciesPolicy_CHEMKIN_WithTransport<SpeciesCHEMKIN>  > PreProcessorSpecies_CHEMKIN;
	typedef OpenSMOKE::PreProcessorKinetics< OpenSMOKE::PreProcessorKineticsPolicy_CHEMKIN<OpenSMOKE::ReactionPolicy_CHEMKIN> > PreProcessorKinetics_CHEMKIN;
	typedef OpenSMOKE::ThermoReader< OpenSMOKE::ThermoReaderPolicy_CHEMKIN< OpenSMOKE::ThermoPolicy_CHEMKIN > > ThermoReader_CHEMKIN;

	std::cout << " * Writing mechanisms..." << std::endl;

	boost::filesystem::path input_path_folder = path_output_folder / "Mechanisms";
	boost::filesystem::path output_path_folder = path_output_folder / "PreprocessedMechanisms";
	if (!boost::filesystem::exists(output_path_folder))
		OpenSMOKE::CreateDirectory(output_path_folder);

	for (unsigned int i = 0; i < nclusters; i++)
	{
		std::cout << "Preprocessing mechanism " << i + 1 << " over " << nclusters << std::endl;

		// Log file
		boost::filesystem::path file_name = output_path_folder / "log";
		std::ofstream flog;
		flog.open(file_name.c_str(), std::ios::out);
		flog.setf(std::ios::scientific);

		std::cout << " * Creating thermo reader..." << std::endl;

		// Readers
		ThermoReader_CHEMKIN* thermoreader;
		OpenSMOKE::TransportReader< OpenSMOKE::TransportReaderPolicy_CHEMKIN<OpenSMOKE::TransportPolicy_CHEMKIN > >* transportreader;

		// Reading thermodynamic database
		thermoreader = new OpenSMOKE::ThermoReader< OpenSMOKE::ThermoReaderPolicy_CHEMKIN< OpenSMOKE::ThermoPolicy_CHEMKIN > >;
		CheckForFatalError(thermoreader->ReadFromFile(chemkin_thermodynamic_file.string()));

		std::cout << " * Creating kinetics reader..." << std::endl;

		// Preprocessing the kinetic mechanism
		std::stringstream label; label << i;
		const std::string local_file_name = "kinetics." + label.str() + ".CKI";
		const boost::filesystem::path chemkin_kinetics_file = input_path_folder / local_file_name;
		PreProcessorKinetics_CHEMKIN preprocessor_kinetics(flog);
		CheckForFatalError(preprocessor_kinetics.ReadFromASCIIFile(chemkin_kinetics_file.string()));

		std::cout << " * Creating transport reader..." << std::endl;

		// Preprocessors
		PreProcessorSpecies_CHEMKIN* preprocessor_species_with_transport = nullptr;

		// Reading transport database
		transportreader = new OpenSMOKE::TransportReader< OpenSMOKE::TransportReaderPolicy_CHEMKIN<OpenSMOKE::TransportPolicy_CHEMKIN > >;
		CheckForFatalError(transportreader->ReadFromFile(chemkin_transport_file.string()));

		// Preprocessing the thermodynamic and transport properties
		preprocessor_species_with_transport = new PreProcessorSpecies_CHEMKIN(*thermoreader, *transportreader, preprocessor_kinetics, flog);
		
		// Preprocessing thermodynamics
		CheckForFatalError(preprocessor_species_with_transport->Setup());
		
		// Read kinetics from file
		CheckForFatalError(preprocessor_kinetics.ReadKineticsFromASCIIFile(preprocessor_species_with_transport->AtomicTable()));

		// Fit transport data
		CheckForFatalError(preprocessor_species_with_transport->Fitting());

		// Write XML files
		std::cout << " * Writing XML files..." << std::endl;
		{
			std::string author_name("undefined");
			std::string place_name("undefined");
			std::string comments("no comments");
			std::string preprocessing_date(OpenSMOKE::GetCurrentDate());
			std::string preprocessing_time(OpenSMOKE::GetCurrentTime());

			std::stringstream xml_string;
			xml_string << std::setprecision(8);
			xml_string.setf(std::ios::scientific);

			xml_string << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
			xml_string << "<opensmoke version=\"0.1a\">" << std::endl;

			xml_string << "<Properties>" << std::endl;
			xml_string << "  <Author>" << author_name << "</Author>" << std::endl;
			xml_string << "  <Place>" << place_name << "</Place>" << std::endl;
			xml_string << "  <Date>" << preprocessing_date << "</Date>" << std::endl;
			xml_string << "  <Time>" << preprocessing_time << "</Time>" << std::endl;
			xml_string << "  <Comments>" << "\n" << OpenSMOKE::SplitStringIntoSeveralLines(comments, 80, "\t\r ") << "\n" << "</Comments>" << std::endl;
			xml_string << "</Properties>" << std::endl;

			// Thermodynamics and transport properties
			preprocessor_species_with_transport->WriteXMLFile(xml_string);

			// Kinetic mechanism
			preprocessor_kinetics.WriteXMLFile(xml_string);

			xml_string << "</opensmoke>" << std::endl;

			// Write file
			std::stringstream label; label << i;
			const std::string folder_name = "kinetics." + label.str();
			const boost::filesystem::path local_folder_path = output_path_folder / folder_name;
			if (!boost::filesystem::exists(local_folder_path))
				OpenSMOKE::CreateDirectory(local_folder_path);

			boost::filesystem::path kinetics_xml = local_folder_path / "kinetics.xml";
			std::ofstream fOutput;
			fOutput.open(std::string(kinetics_xml.string()).c_str(), std::ios::out);
			fOutput.setf(std::ios::scientific);
			fOutput << xml_string.str();
			fOutput.close();
		}

		flog.close();
		delete transportreader;
		delete preprocessor_species_with_transport;
		delete thermoreader;
	}
}
