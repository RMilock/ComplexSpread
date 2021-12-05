# ComplexSpread

Latest Commit of my Master Degree Thesis

The usual simplified description of epidemic dynamics predicts an exponential growth. This is due to the mean field character of the dynamical equations. However, a recent paper (Thurner S, Klimek P and Hanel R 2020 Proc. Nat. Acad. Sci. 117, 22684) \cite{Thurner::NetBasedExpl} showed that in a network with fixed connectivity, the nodes become infected at a rate that increases linearly rather than exponentially.
Experimental data for COVID-19 seem to validate this approach. In this thesis describe the evolution of the COVID-19 pandemic with the SIR model \cite{pizzuti::2020_ItalyCOVIDnetwork}, on different network topologies in order to simulate the different containment policies, i.e. "lock-downs". We will obtain a "sub-exponential" growth of the total cases for a plethora of social networks. This is supported by the presence of a kind of phase transition of the standard deviation of the new cases. In particular, we monitor the effect induced by a significant presence of hubs in the network.
Ultimately, we introduce a parameter that we term the epidemic severity to account for the outbreak size of COVID-19.
