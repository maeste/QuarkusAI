package org.acme.quickstart;

import java.util.HashSet;
import java.util.Set;

import javax.ws.rs.core.Application;

public class LabelApplication extends Application {
	private Set<Object> singletons = new HashSet<Object>();

	public Set<Object> getSingletons() {
		if (singletons.isEmpty()) {
			singletons.add(new GreetingResource());
		}
		return singletons;
	}
}